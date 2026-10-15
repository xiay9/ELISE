#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModel, AutoConfig
import torch.nn as nn
from ..modules.layers import EnhancedLSTM, Biaffine, KERPLEEnhancedLSTM
import torch.nn.functional as F

class SASGEmotionalDynamics(nn.Module):
    """
    SASG (Speaker-Aware and Sentiment-Guided) 情感动态建模

    协同建模跨说话者情感传染与说话者内一致性，对应论文Section 2.4.1-2.4.2
    实现：
    1. 跨说话者情感传染 (Cross-Speaker Emotional Contagion)
    2. 说话者内一致性 (Intra-Speaker Consistency)

    参考论文公式：w^{cont}_{ij} 和一致性分数 C_{ik}
    """
    def __init__(self, emotion_dim, num_speakers=10, alpha=5.0, cfg=None):
        super(SASGEmotionalDynamics, self).__init__()
        self.emotion_dim = emotion_dim

        # 情感影响力预测器 - 计算每个语句的情感影响强度
        drop_rate = getattr(cfg, 'dropout', 0.1) if cfg is not None else 0.1
        self.influence_predictor = nn.Sequential(
            nn.Linear(emotion_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, emotion_dim),
            nn.Tanh()  # 限制影响力范围
        )

        # 说话者易感性嵌入 - 建模个体差异
        self.speaker_susceptibility = nn.Embedding(num_speakers, 1)
        nn.init.uniform_(self.speaker_susceptibility.weight, 0.3, 0.8)  # 易感性在0.3-0.8之间

        # 情感距离计算器 - 计算不同情感类型间的传染概率
        self.emotion_distance_net = nn.Sequential(
            nn.Linear(emotion_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 距离相似度在0-1之间
        )

        # 跨说话者影响调制因子 (从配置读取，有安全默认值)
        self.cross_speaker_strength = getattr(cfg, 'cross_speaker_strength', 0.02)  # 默认保持原值
        self.distance_decay_rate = getattr(cfg, 'distance_decay_rate', 0.5)         # 默认保持原值
        self.influence_window = getattr(cfg, 'window_size', 2)                      # 默认保持原值
        self.enable_contagion = getattr(cfg, 'sasg_contagion', True) if cfg is not None else True
        self.enable_lc = getattr(cfg, 'sasg_lc', True) if cfg is not None else True

        # 预计算距离衰减表（优化3：预计算常量）
        self.register_buffer(
            'decay_table',
            torch.exp(-self.distance_decay_rate * torch.arange(1, self.influence_window + 1, dtype=torch.float32))
        )

        # ==== 说话者内部一致性建模 ====
        self.alpha = alpha  # LC机制的强度参数

        # 说话者内一致性评估器 (对应论文公式 C_{ik})
        self.intra_speaker_consistency = nn.Sequential(
            nn.Linear(emotion_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出0-1的一致性分数
        )

    def forward(self, emotion_features, speaker_ids, utterance_nums, utdg_graph_params=None):
        """
        SASG情感动态建模：协同计算跨说话者传染与说话者内一致性
        """
        batch_size, max_len, emotion_dim = emotion_features.shape
        device = emotion_features.device

        # 计算每个语句的情感影响力（保持原样：不改变dropout/RNG消耗）
        influence_strength = self.influence_predictor(emotion_features)  # [B, L, D]

        # 初始化影响后的特征和相似度矩阵
        influenced_features = emotion_features.clone()
        similarity_matrix = None
        if self.enable_lc:
            init_val = 0.5
            similarity_matrix = emotion_features.new_full((batch_size, max_len, max_len), float(init_val))
            diag_idx = torch.arange(max_len, device=device)
            similarity_matrix[:, diag_idx, diag_idx] = 1.0

        # ====== 优化A：一次性取长度，避免多次 .item() 导致同步 ======
        # 这里不改变任何数值，只减少同步次数
        if utterance_nums.device.type == "cpu":
            utterance_lens = utterance_nums.tolist()
        else:
            utterance_lens = utterance_nums.detach().cpu().tolist()

        # ====== 优化B：一次性计算 graph_lambda（如启用UTDG图参数）======
        graph_lambdas = None
        if utdg_graph_params is not None:
            # [B]，避免每个batch重复softplus
            graph_lambdas = utdg_graph_params.squeeze(-1)

        # 对每个batch单独处理
        for batch_idx, actual_len in enumerate(utterance_lens):

            # 提前终止无效batch（对话长度<=1时无需处理）
            if actual_len <= 1:
                if self.enable_contagion:
                    influenced_features[batch_idx, :actual_len] = emotion_features[batch_idx, :actual_len]
                continue

            # 获取当前batch的有效数据
            batch_features = emotion_features[batch_idx, :actual_len]          # [T, D]
            batch_influence = influence_strength[batch_idx, :actual_len]       # [T, D]
            batch_speakers = speaker_ids[batch_idx, :actual_len]              # [T]

            # 计算说话者易感性
            speaker_susceptibility = torch.sigmoid(
                self.speaker_susceptibility(batch_speakers)
            ).squeeze(-1)                                                     # [T]

            # contagion累计
            cross_speaker_influence = torch.zeros_like(batch_features) if self.enable_contagion else None
            max_offset = min(self.influence_window, actual_len - 1)

            graph_lambda = graph_lambdas[batch_idx] if graph_lambdas is not None else None

            # 预先准备pos（仅用于LC散射索引），避免每次offset生成arange
            pos = torch.arange(actual_len, device=device)

            for offset in range(1, max_offset + 1):
                # ====== 优化C：用slice替代 arange + 高级索引（完全等价）======
                # i_idx = [offset, ..., actual_len-1], j_idx = [0, ..., actual_len-offset-1]
                feat_i = batch_features[offset:]          # == batch_features[i_idx]
                feat_j = batch_features[:-offset]         # == batch_features[j_idx]
                spk_i = batch_speakers[offset:]
                spk_j = batch_speakers[:-offset]

                # 跨说话者情感传染
                if self.enable_contagion:
                    cross_mask = spk_i != spk_j
                    if cross_mask.any():
                        # 注意：cat顺序保持与原实现一致 [feat_j, feat_i]
                        emotion_pair = torch.cat([feat_j, feat_i], dim=-1)
                        emotion_similarity = self.emotion_distance_net(emotion_pair).squeeze(-1)  # [T-offset]

                        if graph_lambda is not None:
                            distance_decay = torch.exp(-graph_lambda * offset)  # scalar
                        else:
                            distance_decay = self.decay_table[offset - 1]       # scalar (buffer already on device)

                        # 等价于 speaker_susceptibility[i_idx] * ...
                        influence_weight = speaker_susceptibility[offset:] * distance_decay * emotion_similarity
                        influence_weight = influence_weight * cross_mask.to(influence_weight.dtype)

                        # 等价于 cross_speaker_influence[i_idx] += ... * batch_influence[j_idx]
                        cross_speaker_influence[offset:] = cross_speaker_influence[offset:] + (influence_weight.unsqueeze(-1) * batch_influence[:-offset])

                # 说话者内一致性（LC）
                if self.enable_lc:
                    same_mask = spk_i == spk_j
                    if same_mask.any():
                        # cat顺序保持与原实现一致 [feat_i, feat_j]
                        emotion_pair = torch.cat([feat_i, feat_j], dim=-1)
                        consistency_score = self.intra_speaker_consistency(emotion_pair).squeeze(-1)  # [T-offset]

                        # i_same/j_same 等价于原来的 i_idx[same_mask], j_idx[same_mask]
                        i_same = pos[offset:][same_mask]
                        j_same = pos[:-offset][same_mask]

                        similarity_matrix[batch_idx, i_same, j_same] = consistency_score[same_mask]
                        similarity_matrix[batch_idx, j_same, i_same] = consistency_score[same_mask]

            # 应用跨说话者影响
            if self.enable_contagion:
                influenced_features[batch_idx, :actual_len] = (
                    batch_features * 0.99 + self.cross_speaker_strength * cross_speaker_influence
                )
            else:
                influenced_features[batch_idx, :actual_len] = batch_features

        # 不启用LC则返回
        if not self.enable_lc:
            return influenced_features, None

        scale = self.alpha * 0.1  # 保守缩放

        # ====== LC后处理：保持原样（不做张量化，避免你之前遇到的更慢/潜在差异）======
        lc_bias = torch.zeros(batch_size, max_len, max_len, 2, device=device, dtype=emotion_features.dtype)
        for batch_idx, actual_len in enumerate(utterance_lens):
            sim = similarity_matrix[batch_idx, :actual_len, :actual_len]
            lc_bias[batch_idx, :actual_len, :actual_len, 1] = scale * (2.0 * sim - 1.0)
        return influenced_features, lc_bias

class UTDGDialogueStatsExtractor(nn.Module):
    """
    UTDG对话统计量提取器 ψ(D)

    提取对话级统计特征，对应论文中的ψ(D)函数
    包括：轮换率、平均话语时长、说话人数量、对话长度等无标签统计量
    """
    def __init__(self):
        super(UTDGDialogueStatsExtractor, self).__init__()

    def forward(self, speaker_ids, utterance_nums):
        """
        提取对话统计量

        Args:
            speaker_ids: [batch_size, max_len] 说话人ID
            utterance_nums: [batch_size] 每个对话的实际长度

        Returns:
            dialogue_stats: [batch_size, stats_dim] 对话统计特征
        """
        batch_size = speaker_ids.shape[0]
        device = speaker_ids.device

        stats_list = []

        for batch_idx in range(batch_size):
            actual_len = utterance_nums[batch_idx].item()
            batch_speakers = speaker_ids[batch_idx, :actual_len]

            # 1. 对话长度 (归一化到0-1)
            dialogue_length = actual_len / 100.0  # 假设最大长度100

            # 2. 说话人数量 (归一化)
            num_speakers = len(torch.unique(batch_speakers[batch_speakers >= 0]))
            num_speakers_norm = num_speakers / 10.0  # 假设最大10个说话人

            # 3. 轮换率 (说话人变化频率)
            if actual_len > 1:
                speaker_changes = (batch_speakers[1:] != batch_speakers[:-1]).sum().float()
                turn_taking_rate = speaker_changes / (actual_len - 1)
            else:
                turn_taking_rate = torch.tensor(0.0, device=device)

            # 4. 说话人分布熵 (多样性)
            speaker_counts = torch.bincount(batch_speakers[batch_speakers >= 0])
            if len(speaker_counts) > 1:
                speaker_probs = speaker_counts.float() / speaker_counts.sum()
                speaker_entropy = -(speaker_probs * torch.log(speaker_probs + 1e-8)).sum()
                speaker_entropy = speaker_entropy / torch.log(torch.tensor(len(speaker_counts), dtype=torch.float32))
            else:
                speaker_entropy = torch.tensor(0.0, device=device)

            # 5. 平均连续发言长度
            continuous_lengths = []
            current_speaker = batch_speakers[0]
            current_length = 1

            for i in range(1, actual_len):
                if batch_speakers[i] == current_speaker:
                    current_length += 1
                else:
                    continuous_lengths.append(current_length)
                    current_speaker = batch_speakers[i]
                    current_length = 1
            continuous_lengths.append(current_length)

            avg_continuous_length = torch.tensor(continuous_lengths, dtype=torch.float32, device=device).mean()
            avg_continuous_length_norm = avg_continuous_length / 10.0  # 归一化

            # 6-8. 补充统计量 (简单的数值特征)
            dialogue_density = torch.tensor(actual_len / (num_speakers + 1e-8), device=device) / 20.0
            first_speaker_dominance = (batch_speakers == batch_speakers[0]).sum().float() / actual_len
            last_speaker_activity = (batch_speakers == batch_speakers[-1]).sum().float() / actual_len

            # 组合统计特征 [8维] - 确保所有元素都是tensor
            stats = torch.stack([
                torch.tensor(dialogue_length, device=device, dtype=torch.float32),
                torch.tensor(num_speakers_norm, device=device, dtype=torch.float32),
                turn_taking_rate.to(device),
                speaker_entropy.to(device),
                avg_continuous_length_norm.to(device),
                dialogue_density.to(device),
                first_speaker_dominance.to(device),
                last_speaker_activity.to(device)
            ])

            stats_list.append(stats)

        # [batch_size, stats_dim]
        dialogue_stats = torch.stack(stats_list, dim=0)
        return dialogue_stats

class UTDGUnifiedGenerator(nn.Module):
    """
    UTDG统一时间衰减生成器

    对应论文中的核心创新：
    1. f_φ: 对话统计量 → 潜变量z
    2. g_seq(z): 潜变量z → 序列通路核参数
    3. g_graph(z): 潜变量z → 图通路核参数

    实现同源一致的时序归纳偏置生成
    """
    def __init__(self, stats_dim=8, d_z=32, hidden_dim=64, num_modalities=3, dropout=0.1):
        super(UTDGUnifiedGenerator, self).__init__()

        self.d_z = d_z
        self.num_modalities = num_modalities

        # 对话统计量提取器 ψ(D)
        self.stats_extractor = UTDGDialogueStatsExtractor()

        # 潜变量生成器 f_φ: 论文中的两层MLP (256→64→32)
        # 这里保守设置为 stats_dim→hidden_dim→d_z
        self.latent_generator = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_z)
        )

        # 序列通路核参数映射器 g_seq(z)（固定 exponential: decay_rates）
        self.g_seq = nn.Linear(d_z, num_modalities)  # → decay_rates

        # 图通路核参数映射器 g_graph(z) (简化：单一衰减核)
        self.g_graph = nn.Linear(d_z, 1)  # → single decay parameter for graph

    def forward(self, speaker_ids, utterance_nums):
        """
        UTDG前向传播

        Args:
            speaker_ids: [batch_size, max_len] 说话人ID
            utterance_nums: [batch_size] 对话实际长度

        Returns:
            seq_kernel_params: 序列通路核参数
            graph_kernel_params: 图通路核参数
            latent_z: 潜变量z (用于对齐正则化)
        """
        # 1. 提取对话统计量 ψ(D)
        dialogue_stats = self.stats_extractor(speaker_ids, utterance_nums)

        # 2. 生成潜变量 z = f_φ(ψ(D))
        latent_z = self.latent_generator(dialogue_stats)  # [batch_size, d_z]

        # 3. 生成序列通路核参数 κ_seq(d) = κ(d; g_seq(z))
        seq_params_raw = self.g_seq(latent_z)  # [batch_size, param_dim]

        # 4. 生成图通路核参数 κ_graph(d) = κ(d; g_graph(z))
        graph_params_raw = self.g_graph(latent_z)  # [batch_size, 1]

        # 5. 参数后处理（固定 exponential）
        seq_kernel_params = F.softplus(seq_params_raw) * 0.5 + 0.05    # [B, M]
        graph_kernel_params = F.softplus(graph_params_raw) * 0.3 + 0.1  # [B, 1]

        return seq_kernel_params, graph_kernel_params, latent_z

class PaceExtModel(nn.Module):
    def __init__(self, cfg, tokenizer):
        super(PaceExtModel, self).__init__()
        self.cfg = cfg
        self.enable_sasg = getattr(cfg, 'sasg_enable', True)
        self.enable_kerple = getattr(cfg, 'enable_kerple', True)
        self.mm_fuse_learn = getattr(cfg, 'mm_fuse_learn', True)
        self.pair_confdiff_enable = bool(getattr(cfg, 'pair_confdiff_enable', False))
        self.pair_confdiff_margin = float(getattr(cfg, 'pair_confdiff_margin', 0.05))
        self.pair_confdiff_loss_weight = float(getattr(cfg, 'pair_confdiff_loss_weight', 0.0))
        self.pair_confdiff_dynamic_margin = bool(getattr(cfg, 'pair_confdiff_dynamic_margin', True))
        self.pair_confdiff_detach_node_confidence = bool(getattr(cfg, 'pair_confdiff_detach_node_confidence', True))
        self.pair_confdiff_row_topk_negatives = max(0, int(getattr(cfg, 'pair_confdiff_row_topk_negatives', 0)))
        self.pair_confdiff_warmup_epochs = max(0, int(getattr(cfg, 'pair_confdiff_warmup_epochs', 0)))
        self.pair_confdiff_ramp_epochs = max(1, int(getattr(cfg, 'pair_confdiff_ramp_epochs', 1)))
        self.pair_robust_enable = bool(getattr(cfg, 'pair_robust_enable', False))
        self.pair_robust_utterance_drop_prob = float(getattr(cfg, 'pair_robust_utterance_drop_prob', 0.0))
        self.pair_robust_noise_std = float(getattr(cfg, 'pair_robust_noise_std', 0.0))
        self.pair_robust_protect_gold_utts = bool(getattr(cfg, 'pair_robust_protect_gold_utts', True))
        self.pair_robust_corrupt_loss_weight = float(getattr(cfg, 'pair_robust_corrupt_loss_weight', 0.0))
        self.pair_robust_consistency_weight = float(getattr(cfg, 'pair_robust_consistency_weight', 0.0))

        if (self.pair_confdiff_enable or self.pair_robust_enable) and (
            bool(getattr(cfg, 'pair_role_enable', False))
            or str(getattr(cfg, 'context_subtype_task', 'none')).lower() not in ('none', '')
        ):
            raise ValueError('RPCL losses support binary ECPE only; disable them for pair-role/subtype tasks.')

        bert_config = AutoConfig.from_pretrained(cfg.backbone)
        self.tokenizer = tokenizer

        num_classes = 7 if getattr(cfg, 'emo_cat', 'yes') == 'yes' else 2
        num = 2  # LSTM双向拼接

        drop_rate = getattr(cfg, 'dropout', 0.1)

        self.video_linear = nn.Sequential(
            nn.Linear(cfg.video_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.audio_linear = nn.Sequential(
            nn.Linear(cfg.audio_dim, bert_config.hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(bert_config.hidden_size, bert_config.hidden_size),
        )

        self.emotion_linear = nn.Sequential(
            nn.Linear(bert_config.hidden_size * num, cfg.hid_size),
            nn.ReLU(),
            nn.Linear(cfg.hid_size, num_classes)
        )
        self.cause_linear = nn.Linear(bert_config.hidden_size * num, 2)
        self.context_subtype_enable = bool(getattr(cfg, 'context_subtype_enable', False))
        self.context_subtype_num_classes = int(getattr(cfg, 'context_subtype_num_classes', 0))
        self.pair_num_classes = int(getattr(cfg, 'pair_num_classes', 2))
        self.biaffine = Biaffine(bert_config.hidden_size * num, self.pair_num_classes)
        if self.context_subtype_enable:
            self.context_subtype_biaffine = Biaffine(
                bert_config.hidden_size * num,
                self.context_subtype_num_classes,
            )

        # Dozat-style MLP projection before biaffine (head/tail)
        pair_dim = bert_config.hidden_size * num
        self.pair_mlp_head = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )
        self.pair_mlp_tail = nn.Sequential(
            nn.Linear(pair_dim, pair_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

        # 使用KERPLE-Lite增强的LSTM架构（每个模态独立）
        self.text_kerple_lstm = KERPLEEnhancedLSTM(
            input_size=bert_config.hidden_size,
            hidden_size=bert_config.hidden_size,
            num_layers=1,
            max_len=getattr(cfg, 'kerple_max_len', 100),
            ff_dropout=drop_rate,
            recurrent_dropout=drop_rate,
            bidirectional=True
        )
        self.audio_kerple_lstm = KERPLEEnhancedLSTM(
            input_size=bert_config.hidden_size,
            hidden_size=bert_config.hidden_size,
            num_layers=1,
            max_len=getattr(cfg, 'kerple_max_len', 100),
            ff_dropout=drop_rate,
            recurrent_dropout=drop_rate,
            bidirectional=True
        )
        self.video_kerple_lstm = KERPLEEnhancedLSTM(
            input_size=bert_config.hidden_size,
            hidden_size=bert_config.hidden_size,
            num_layers=1,
            max_len=getattr(cfg, 'kerple_max_len', 100),
            ff_dropout=drop_rate,
            recurrent_dropout=drop_rate,
            bidirectional=True
        )

        # KERPLE融合权重参数
        self.kerple_alpha = getattr(cfg, 'kerple_alpha', 0.7)

        # 多模态KERPLE融合的可学习凸权重 ω（text, audio, video），经softmax保证为凸组合
        self.mm_fuse_weights = nn.Parameter(torch.ones(3))
        if not self.mm_fuse_learn:
            self.mm_fuse_weights.requires_grad_(False)

        # 保留原始LSTM作为备用（用于对比实验）
        self.enhanced_lstm = EnhancedLSTM(
            bert_config.hidden_size,
            bert_config.hidden_size,
            1,
            ff_dropout=drop_rate,
            recurrent_dropout=drop_rate,
            bidirectional=True
        )

        # SASG情感动态建模: 跨说话者传染 + 说话者内一致性
        self.sasg_dynamics = SASGEmotionalDynamics(
            emotion_dim=bert_config.hidden_size,
            num_speakers=len(cfg.speaker_dict),
            alpha=cfg.alpha,
            cfg=cfg  # 传递配置对象，用于读取窗口大小等参数
        )

        # UTDG统一时间衰减生成器 (安全opt-in设计)
        utdg_mode = getattr(cfg, 'utdg_mode', None)
        if utdg_mode is None:
            self.enable_utdg = getattr(cfg, 'enable_utdg', False)
            self.enable_utdg_seq = self.enable_utdg
            self.enable_utdg_graph = self.enable_utdg
        else:
            self.enable_utdg = utdg_mode != 'off'
            self.enable_utdg_seq = utdg_mode in ('full', 'seq')
            self.enable_utdg_graph = utdg_mode in ('full', 'graph')

        if self.enable_utdg:
            self.utdg_generator = UTDGUnifiedGenerator(
                stats_dim=getattr(cfg, 'utdg_stats_dim', 8),
                d_z=getattr(cfg, 'd_z', 32),
                hidden_dim=getattr(cfg, 'utdg_hidden_dim', 64),
                num_modalities=3,
                dropout=drop_rate
            )

        self.apply(self._init_esim_weights)

        self.bert = AutoModel.from_pretrained(cfg.backbone)

    def process_multimodal_kerple(self, text_encoded, audio_encoded, video_encoded, fused_input, utterance_nums, utdg_seq_params=None):
        """
        多模态KERPLE-Lite处理：为每个模态应用不同的核衰减，然后融合
        支持UTDG生成的动态核参数
        """
        if not self.enable_kerple:
            fused_baseline = text_encoded + audio_encoded + video_encoded
            return self.enhanced_lstm(fused_baseline, None, utterance_nums.cpu())

        # 使用传统LSTM处理融合后的特征作为基准（保持 enable_kerple=True 时原逻辑不变）
        base_output = self.enhanced_lstm(fused_input, None, utterance_nums.cpu())

        # 为每个模态使用独立的KERPLE-Lite LSTM
        utdp_params = utdg_seq_params
        text_kerple = self.text_kerple_lstm(
            text_encoded, utterance_nums, modality_id=0,
            utdp_params=utdp_params
        )
        audio_kerple = self.audio_kerple_lstm(
            audio_encoded, utterance_nums, modality_id=1,
            utdp_params=utdp_params
        )
        video_kerple = self.video_kerple_lstm(
            video_encoded, utterance_nums, modality_id=2,
            utdp_params=utdp_params
        )

        # 多模态KERPLE融合：可学习凸权 ω 对三模态进行加权组合
        if self.mm_fuse_learn:
            mm_w = torch.softmax(self.mm_fuse_weights, dim=0)  # [3], 和为1
        else:
            mm_w = torch.full_like(self.mm_fuse_weights, 1.0 / 3.0)
        kerple_combined = (
            mm_w[0] * text_kerple +
            mm_w[1] * audio_kerple +
            mm_w[2] * video_kerple
        )

        # 自适应融合：学习如何平衡KERPLE增强和传统方法
        final_output = self.kerple_alpha * kerple_combined + (1 - self.kerple_alpha) * base_output

        return final_output

    def _init_esim_weights(self, module):
        """
        Initialise the weights of the ESIM model.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_utt_mask(self, input, utterance_nums, pair_nums, pairs):
        mask = torch.arange(input.shape[1]).unsqueeze(0).to(input.device) < utterance_nums.unsqueeze(-1)
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        triu = torch.flip(torch.flip(torch.triu(torch.ones_like(mask[0])), [1]), [0])
        for i in range(len(utterance_nums)):
            mask[i] = mask[i] * triu

        batch_size, seq_len  = input.shape[:2]

        gold = input.new_zeros((batch_size, seq_len, seq_len), dtype=torch.long)
        for i in range(len(input)):
            if pair_nums[i] == 0:
                continue
            gold[i, [w[0] for w in pairs[i, :pair_nums[i]]], [w[1] for w in pairs[i, :pair_nums[i]]]] = 1
        return mask, gold

    def get_pair_scores(self, input, biaffine=None):
        biaffine = self.biaffine if biaffine is None else biaffine
        head = self.pair_mlp_head(input)
        tail = self.pair_mlp_tail(input)
        product = biaffine(head, tail).squeeze(-1)
        if len(product.shape) == 3:
            product = product.unsqueeze(-1)
        return product.transpose(2, 1).transpose(3, 2).contiguous()

    def _sample_pair_role_loss_items(self, logits, gold):
        if not self.training:
            return logits, gold
        strategy = str(getattr(self.cfg, 'pair_role_pair_sampling', 'none')).strip().lower()
        if strategy in ('', 'none', 'off', 'false'):
            return logits, gold

        num_classes = int(logits.shape[-1])
        groups = []
        for class_id in range(num_classes):
            class_indices = torch.nonzero(gold.long() == class_id, as_tuple=False).flatten()
            if class_indices.numel() > 0:
                groups.append((class_id, class_indices))
        if len(groups) < int(getattr(self.cfg, 'pair_role_pair_sampling_min_classes', 2)):
            return logits, gold

        context_id = int(getattr(self.cfg, 'pair_role_context_class_id', 1))
        if bool(getattr(self.cfg, 'pair_role_pair_sampling_require_context', True)):
            if not any(class_id == context_id for class_id, _ in groups):
                return logits, gold

        if strategy in ('context_oversample', 'context-over', 'oversample_context'):
            factor = float(getattr(self.cfg, 'pair_role_pair_context_oversample_factor', 1.0))
            context_indices = None
            for class_id, class_indices in groups:
                if class_id == context_id:
                    context_indices = class_indices
                    break
            if context_indices is None or context_indices.numel() == 0 or factor <= 1.0:
                return logits, gold

            sampled = [torch.arange(gold.numel(), device=gold.device)]
            full_extra = max(0, int(factor) - 1)
            if full_extra > 0:
                sampled.extend([context_indices] * full_extra)
            fractional_extra = factor - int(factor)
            if fractional_extra > 0.0:
                extra_count = int(round(float(context_indices.numel()) * fractional_extra))
                if extra_count > 0:
                    sampled.append(context_indices[torch.randint(context_indices.numel(), (extra_count,), device=gold.device)])
            selected = torch.cat(sampled)
        elif strategy in ('balanced', 'balance', 'balanced_pair', 'pair_balanced'):
            counts = sorted(int(indices.numel()) for _, indices in groups)
            configured_target = int(getattr(self.cfg, 'pair_role_pair_balance_target_per_class', 0))
            if configured_target > 0:
                target = configured_target
            else:
                target = counts[len(counts) // 2]
            min_target = int(getattr(self.cfg, 'pair_role_pair_balance_min_per_class', 1))
            max_target = int(getattr(self.cfg, 'pair_role_pair_balance_max_per_class', 0))
            target = max(min_target, int(target))
            if max_target > 0:
                target = min(target, max_target)

            sampled = []
            for _, class_indices in groups:
                if class_indices.numel() >= target:
                    sampled.append(class_indices[torch.randperm(class_indices.numel(), device=gold.device)[:target]])
                else:
                    sampled.append(class_indices[torch.randint(class_indices.numel(), (target,), device=gold.device)])
            selected = torch.cat(sampled)
        else:
            return logits, gold

        max_items = int(getattr(self.cfg, 'pair_role_pair_sampling_max_items', 0))
        if max_items > 0 and selected.numel() > max_items:
            selected = selected[torch.randperm(selected.numel(), device=gold.device)[:max_items]]
        selected = selected[torch.randperm(selected.numel(), device=gold.device)]
        return logits[selected], gold[selected]

    def get_dot_product(self, input, masks, gold_matrix, similarity=None, class_labels=None):
        # input: batch_size, max_utterance_num, hidden_dim
        # utterance_nums: batch_size
        # pairs: batch_size, 2
        product = self.get_pair_scores(input)

        # Apply label-constraint similarity to the positive/emo-cause channel only.
        if similarity is not None:
            product = product.clone()
            if similarity.dim() == product.dim() and similarity.shape[-1] == product.shape[-1]:
                product = product + similarity
            else:
                if similarity.dim() == product.dim() and similarity.shape[-1] >= 2:
                    similarity_bias = similarity[..., 1]
                else:
                    similarity_bias = similarity
                if product.shape[-1] == 2:
                    target_class = 1
                else:
                    target_class = int(getattr(self.cfg, 'pair_role_cause_class_id', 0))
                product[..., target_class] = product[..., target_class] + similarity_bias

        self._last_pair_role_main_loss = product.new_zeros(())
        self._last_pair_role_related_loss = product.new_zeros(())

        if class_labels is None:
            active_labels = gold_matrix
            activate_loss = masks.reshape(-1) == 1
            class_weights = torch.tensor(
                [1.0, float(getattr(self.cfg, 'loss_weight', 2.0))],
                dtype=torch.float32,
                device=input.device,
            )
        else:
            ignore_index = int(getattr(self.cfg, 'pair_role_ignore_index', -100))
            active_labels = class_labels.to(input.device)
            activate_loss = (masks.reshape(-1) == 1) & (active_labels.reshape(-1) != ignore_index)
            configured_weights = getattr(self.cfg, 'pair_role_class_weights', None)
            if configured_weights is not None and len(configured_weights) == int(product.shape[-1]):
                class_weights = torch.tensor(configured_weights, dtype=torch.float32, device=input.device)
            else:
                class_weights = None

        if not bool(activate_loss.any()):
            return product.new_zeros(()), product

        activate_logits = product.reshape(-1, product.shape[-1])[activate_loss]
        activate_gold = active_labels.reshape(-1)[activate_loss]
        if class_labels is not None:
            activate_logits, activate_gold = self._sample_pair_role_loss_items(activate_logits, activate_gold)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        pair_role_loss = criterion(activate_logits, activate_gold.long())
        loss = pair_role_loss
        self._last_pair_role_main_loss = pair_role_loss.detach()

        related_loss_weight = float(getattr(self.cfg, 'pair_role_related_loss_weight', 0.0))
        if class_labels is not None and related_loss_weight > 0.0 and product.shape[-1] >= 3:
            cause_id = int(getattr(self.cfg, 'pair_role_cause_class_id', 0))
            context_id = int(getattr(self.cfg, 'pair_role_context_class_id', 1))
            non_pair_id = int(getattr(self.cfg, 'pair_role_non_pair_class_id', 2))
            related_ids = [
                class_id
                for class_id in (cause_id, context_id)
                if 0 <= int(class_id) < int(product.shape[-1]) and int(class_id) != non_pair_id
            ]
            if related_ids and 0 <= non_pair_id < int(product.shape[-1]):
                related_logit = torch.logsumexp(activate_logits[:, related_ids], dim=-1)
                non_pair_logit = activate_logits[:, non_pair_id]
                related_logits = torch.stack([related_logit, non_pair_logit], dim=-1)
                related_gold = torch.where(
                    activate_gold.long() == non_pair_id,
                    torch.ones_like(activate_gold.long()),
                    torch.zeros_like(activate_gold.long()),
                )
                configured_related_weights = getattr(self.cfg, 'pair_role_related_class_weights', None)
                related_weights = None
                if configured_related_weights is not None and len(configured_related_weights) == 2:
                    related_weights = torch.tensor(configured_related_weights, dtype=torch.float32, device=input.device)
                related_criterion = nn.CrossEntropyLoss(weight=related_weights)
                related_loss = related_criterion(related_logits, related_gold)
                self._last_pair_role_related_loss = related_loss.detach()
                loss = loss + (related_loss_weight * related_loss)
        if torch.isnan(loss):
            loss = 0
        return loss, product

    def merge_input(self, input, indices):
        """Merge token-level outputs into utterance-level representations."""

        max_utterance_num = max([len(w) for w in indices])

        res = input.new_zeros((len(indices), max_utterance_num, input.shape[-1]))

        for i in range(len(indices)):
            cur_id = indices[i][0][0]
            end_id = indices[i][-1][0]
            cur_lens = 0
            for j in range(cur_id, end_id + 1):
                start = input.new_tensor([w[1] for w in indices[i] if w[0] == j], dtype=torch.long)
                end = input.new_tensor([w[2] - 1 for w in indices[i] if w[0] == j], dtype=torch.long)
                start_rep = torch.gather(input[j], 0, start.unsqueeze(-1).expand(-1, input.shape[-1]))
                end_rep = torch.gather(input[j], 0, end.unsqueeze(-1).expand(-1, input.shape[-1]))

                end = input.new_tensor([w[2] for w in indices[i] if w[0] == j], dtype=torch.long)

                lens = start.shape[0]
                res[i, cur_lens:cur_lens + lens] = end_rep + input[j][0].unsqueeze(0)
                cur_lens += lens
        return res

    def get_emotion(self, logits, utterance_nums, emotion_labels, emo=True):
        mask = torch.arange(logits.shape[1]).unsqueeze(0).to(logits.device) < utterance_nums.unsqueeze(-1)
        mask = mask.to(logits.device)
        activate_loss = mask.view(-1) == 1
        activate_logits = logits.view(-1, logits.shape[-1])[activate_loss]

        activate_gold = emotion_labels.view(-1)[activate_loss]
        if getattr(self.cfg, 'emo_cat', 'yes') == 'yes' and emo:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0] + [1.5] * 6).to(logits.device))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(logits.device))
        loss = criterion(activate_logits, activate_gold.long())
        return loss

    def get_context_subtype_loss(self, logits, labels=None, mask=None):
        if labels is None or mask is None:
            return logits.new_zeros(())
        labels = labels.to(logits.device).float()
        mask = mask.to(logits.device).float()
        active = mask > 0
        if not bool(active.any()):
            return logits.new_zeros(())
        configured_weights = getattr(self.cfg, 'context_subtype_pos_weights', None)
        pos_weight = None
        if configured_weights is not None and len(configured_weights) == int(logits.shape[-1]):
            pos_weight = torch.tensor(configured_weights, dtype=torch.float32, device=logits.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        loss = criterion(logits, labels)
        loss = loss * mask.unsqueeze(-1)
        denom = (mask.sum() * logits.shape[-1]).clamp_min(1.0)
        loss = loss.sum() / denom
        if torch.isnan(loss):
            loss = logits.new_zeros(())
        return loss

    def _pair_positive_class_id(self, ecp_logits):
        if ecp_logits.shape[-1] == 2 and not bool(getattr(self.cfg, 'pair_role_enable', False)):
            return 1
        return int(getattr(self.cfg, 'pair_role_cause_class_id', 0))

    def get_teacher_pair_ranking_loss(
        self,
        ecp_logits,
        ecp_mask,
        teacher_pair_positive_mask=None,
        teacher_pair_negative_mask=None,
    ):
        if teacher_pair_positive_mask is None or teacher_pair_negative_mask is None:
            return ecp_logits.new_zeros(())

        pair_positive_logits = ecp_logits[..., self._pair_positive_class_id(ecp_logits)]
        valid_mask = ecp_mask.to(ecp_logits.device).bool()
        positive_weights_full = teacher_pair_positive_mask.to(ecp_logits.device) * valid_mask.float()
        negative_weights_full = teacher_pair_negative_mask.to(ecp_logits.device) * valid_mask.float()
        positive_mask = positive_weights_full > 0
        negative_mask = negative_weights_full > 0

        row_losses = []
        batch_size, max_rows = positive_mask.shape[:2]
        for batch_index in range(batch_size):
            for row_index in range(max_rows):
                pos_logits = pair_positive_logits[batch_index, row_index][positive_mask[batch_index, row_index]]
                neg_logits = pair_positive_logits[batch_index, row_index][negative_mask[batch_index, row_index]]
                pos_weights = positive_weights_full[batch_index, row_index][positive_mask[batch_index, row_index]]
                neg_weights = negative_weights_full[batch_index, row_index][negative_mask[batch_index, row_index]]
                if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                    continue
                score_diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
                pair_weights = pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
                pair_loss = -F.logsigmoid(score_diff)
                denom = pair_weights.sum().clamp_min(1e-12)
                row_loss = (pair_loss * pair_weights).sum() / denom
                if torch.isnan(row_loss):
                    continue
                row_losses.append(row_loss)

        if not row_losses:
            return ecp_logits.new_zeros(())
        return torch.stack(row_losses).mean()

    def get_teacher_loss_weight(self) -> float:
        target_weight = float(getattr(self.cfg, 'teacher_ranking_loss_weight', 0.0))
        if not bool(getattr(self.cfg, 'teacher_distill_enable', False)):
            return 0.0
        if not bool(getattr(self.cfg, 'teacher_rank_curriculum_enable', False)):
            return target_weight

        warmup_epochs = max(0, int(getattr(self.cfg, 'teacher_rank_warmup_epochs', 0)))
        ramp_epochs = max(1, int(getattr(self.cfg, 'teacher_rank_ramp_epochs', 1)))
        current_epoch = int(getattr(self, 'global_epoch', 0))
        if current_epoch < warmup_epochs:
            return 0.0

        progress_epoch = current_epoch - warmup_epochs + 1
        factor = min(1.0, max(0.0, float(progress_epoch) / float(ramp_epochs)))
        return float(target_weight * factor)

    def get_pair_confdiff_loss_weight(self) -> float:
        if not self.pair_confdiff_enable:
            return 0.0
        target_weight = float(self.pair_confdiff_loss_weight)
        current_epoch = int(getattr(self, 'global_epoch', 0))
        if current_epoch < self.pair_confdiff_warmup_epochs:
            return 0.0

        progress_epoch = current_epoch - self.pair_confdiff_warmup_epochs + 1
        factor = min(1.0, max(0.0, float(progress_epoch) / float(self.pair_confdiff_ramp_epochs)))
        return float(target_weight * factor)

    def get_pair_confdiff_ranking_loss(
        self,
        ecp_logits,
        ecp_mask,
        gold_matrix,
        emotion_logits,
        cause_logits,
    ):
        if not self.pair_confdiff_enable:
            return ecp_logits.new_zeros(()), 0.0

        valid_mask = ecp_mask.to(ecp_logits.device).bool()
        pair_scores = ecp_logits[..., 1]
        gold_mask = gold_matrix.to(ecp_logits.device).bool() & valid_mask
        negative_mask = (~gold_mask) & valid_mask

        if getattr(self.cfg, 'emo_cat', 'yes') == 'yes':
            emotion_prob = torch.softmax(emotion_logits, dim=-1)
            neutral_prob = emotion_prob[..., 0]
            emo_conf = 1.0 - neutral_prob
        else:
            emo_conf = torch.softmax(emotion_logits, dim=-1)[..., 1]
        cause_conf = torch.softmax(cause_logits, dim=-1)[..., 1]

        if self.pair_confdiff_detach_node_confidence:
            emo_conf = emo_conf.detach()
            cause_conf = cause_conf.detach()

        row_losses = []
        total_negatives = 0
        selected_negatives = 0
        batch_size, max_rows = pair_scores.shape[:2]
        for batch_index in range(batch_size):
            for row_index in range(max_rows):
                row_pos_mask = gold_mask[batch_index, row_index]
                row_neg_mask = negative_mask[batch_index, row_index]
                if not row_pos_mask.any() or not row_neg_mask.any():
                    continue

                pos_scores = pair_scores[batch_index, row_index][row_pos_mask]
                pos_cause_conf = cause_conf[batch_index][row_pos_mask]
                pos_row_conf = emo_conf[batch_index, row_index]
                pos_confdiff = pos_row_conf.unsqueeze(0) - pos_cause_conf

                neg_scores_full = pair_scores[batch_index, row_index][row_neg_mask]
                neg_cause_conf_full = cause_conf[batch_index][row_neg_mask]
                neg_row_conf = emo_conf[batch_index, row_index]
                neg_confdiff_full = neg_row_conf.unsqueeze(0) - neg_cause_conf_full
                total_negatives += int(neg_scores_full.numel())

                if (
                    self.pair_confdiff_row_topk_negatives > 0
                    and neg_scores_full.numel() > self.pair_confdiff_row_topk_negatives
                ):
                    topk_scores, topk_indices = torch.topk(
                        neg_scores_full,
                        k=self.pair_confdiff_row_topk_negatives,
                        largest=True,
                    )
                    neg_scores = topk_scores
                    neg_confdiff = neg_confdiff_full[topk_indices]
                else:
                    neg_scores = neg_scores_full
                    neg_confdiff = neg_confdiff_full
                selected_negatives += int(neg_scores.numel())

                score_margin = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                if self.pair_confdiff_dynamic_margin:
                    conf_margin = pos_confdiff.unsqueeze(1) - neg_confdiff.unsqueeze(0)
                    target_margin = self.pair_confdiff_margin + conf_margin
                else:
                    target_margin = self.pair_confdiff_margin
                pair_loss = F.relu(target_margin - score_margin)
                row_loss = pair_loss.mean()
                if torch.isnan(row_loss):
                    continue
                row_losses.append(row_loss)

        if not row_losses:
            return ecp_logits.new_zeros(()), 0.0

        selected_ratio = 1.0 if total_negatives == 0 else float(selected_negatives / max(total_negatives, 1))
        return torch.stack(row_losses).mean(), selected_ratio

    def build_pair_robust_mask(self, utterance_nums, pairs, pair_nums, max_len, device):
        if self.pair_robust_utterance_drop_prob <= 0.0:
            return torch.zeros((len(utterance_nums), max_len), dtype=torch.bool, device=device)

        valid_mask = torch.arange(max_len, device=device).unsqueeze(0) < utterance_nums.unsqueeze(-1)
        corruptible_mask = valid_mask.clone()
        if self.pair_robust_protect_gold_utts:
            protected_mask = torch.zeros_like(corruptible_mask)
            batch_size = len(utterance_nums)
            for batch_index in range(batch_size):
                if pair_nums[batch_index] == 0:
                    continue
                for row_index, col_index in pairs[batch_index, : pair_nums[batch_index]]:
                    if 0 <= row_index < max_len:
                        protected_mask[batch_index, row_index] = True
                    if 0 <= col_index < max_len:
                        protected_mask[batch_index, col_index] = True
            corruptible_mask = corruptible_mask & (~protected_mask)

        corruption_mask = (
            torch.rand(corruptible_mask.shape, device=device) < self.pair_robust_utterance_drop_prob
        ) & corruptible_mask
        for batch_index in range(corruptible_mask.shape[0]):
            if corruption_mask[batch_index].any() or not corruptible_mask[batch_index].any():
                continue
            candidate_indices = torch.nonzero(corruptible_mask[batch_index], as_tuple=False).view(-1)
            chosen_index = candidate_indices[torch.randint(len(candidate_indices), (1,), device=device)]
            corruption_mask[batch_index, chosen_index] = True
        return corruption_mask

    def apply_pair_robust_corruption(self, features, corruption_mask):
        if features is None or not corruption_mask.any():
            return features
        expanded_mask = corruption_mask.unsqueeze(-1)
        corrupted = torch.where(expanded_mask, torch.zeros_like(features), features)
        if self.pair_robust_noise_std > 0.0:
            noise = torch.randn_like(features) * self.pair_robust_noise_std
            corrupted = torch.where(expanded_mask, corrupted + noise, corrupted)
        return corrupted

    def get_pair_robust_consistency_loss(self, clean_logits, corrupted_logits, ecp_mask):
        activate_loss = ecp_mask.view(-1) == 1
        if not activate_loss.any():
            return clean_logits.new_zeros(())
        clean_pair = torch.softmax(clean_logits.view(-1, 2)[activate_loss], dim=-1)
        corrupted_pair = torch.softmax(corrupted_logits.view(-1, 2)[activate_loss], dim=-1)
        return F.mse_loss(corrupted_pair, clean_pair.detach())

    def forward(self, **kwargs):
        input_ids, input_masks, utterance_nums = [kwargs[w] for w in 'input_ids input_masks utterance_nums'.split()]
        pairs, pair_nums, labels, indices = [kwargs[w] for w in 'pairs pair_nums labels indices'.split()]
        cause_labels, speaker_ids = [kwargs[w] for w in ['cause_labels', 'speaker_ids']]
        audio_features, video_features = [kwargs[w] for w in ['audio_features', 'video_features']]
        teacher_pair_positive_mask = kwargs.get('teacher_pair_positive_mask')
        teacher_pair_negative_mask = kwargs.get('teacher_pair_negative_mask')
        pair_role_labels = kwargs.get('pair_role_labels', kwargs.get('verifiability_labels'))
        context_subtype_labels = kwargs.get('context_subtype_labels')
        context_subtype_mask = kwargs.get('context_subtype_mask')

        input = self.bert(input_ids, attention_mask=input_masks)[0]

        text = self.merge_input(input, indices)
        if not bool(getattr(self.cfg, 'use_text', True)):
            text = torch.zeros_like(text)
        audio = self.audio_linear(audio_features)
        video = self.video_linear(video_features)

        # 不注入speaker
        text_with_speaker = text

        text_encoded, audio_encoded, video_encoded = text_with_speaker, audio, video

        # UTDG统一时间衰减生成器（可选，需在SASG之前以便注入图通路核）
        utdg_seq_params, utdg_graph_params, utdg_latent_z = None, None, None
        if self.enable_utdg:
            utdg_seq_params, utdg_graph_params, utdg_latent_z = self.utdg_generator(speaker_ids, utterance_nums)
            if not self.enable_utdg_seq:
                utdg_seq_params = None
            if not self.enable_utdg_graph:
                utdg_graph_params = None

        # SASG情感动态建模: 跨说话者传染 + 说话者内一致性协同（注入UTDG图通路衰减）
        if self.enable_sasg:
            influenced_input, similarity_matrix = self.sasg_dynamics(
                text_with_speaker, speaker_ids, utterance_nums, utdg_graph_params
            )
        else:
            influenced_input, similarity_matrix = text_with_speaker, None

        # 方案C：contagion 只影响 Emotion/Cause；LC 只影响 Pair
        # Pair 分支始终使用未受 contagion 影响的文本特征
        multimodal_output_pair = self.process_multimodal_kerple(
            text_encoded, audio_encoded, video_encoded, text_with_speaker, utterance_nums, utdg_seq_params
        )

        # Emotion/Cause 分支仅在 contagion 开启时使用增强后的文本特征，否则复用 Pair 分支结果以省算力
        if self.enable_sasg and getattr(self.sasg_dynamics, 'enable_contagion', False):
            multimodal_output_emc = self.process_multimodal_kerple(
                influenced_input, audio_encoded, video_encoded, influenced_input, utterance_nums, utdg_seq_params
            )
        else:
            multimodal_output_emc = multimodal_output_pair

        # 使用对应分支特征进行预测
        emotion_logits = self.emotion_linear(multimodal_output_emc)
        emo_loss = self.get_emotion(emotion_logits, utterance_nums, labels, emo=True)

        cause_logits = self.cause_linear(multimodal_output_emc)
        cause_loss = self.get_emotion(cause_logits, utterance_nums, cause_labels, emo=False)
        context_subtype_logits = None
        context_subtype_loss = cause_logits.new_zeros(())

        ecp_mask, gold_matrix = self.get_utt_mask(multimodal_output_pair, utterance_nums, pair_nums, pairs)

        # 应用SASG生成的说话者内一致性相似度矩阵
        use_pair_role = bool(getattr(self.cfg, 'pair_role_enable', False))
        ecp_loss, ecp_logits = self.get_dot_product(
            multimodal_output_pair,
            ecp_mask,
            gold_matrix,
            similarity_matrix,
            class_labels=pair_role_labels if use_pair_role else None,
        )
        if self.context_subtype_enable:
            context_subtype_logits = self.get_pair_scores(
                multimodal_output_pair,
                self.context_subtype_biaffine,
            )
            context_subtype_loss = self.get_context_subtype_loss(
                context_subtype_logits,
                context_subtype_labels,
                context_subtype_mask,
            )
        pair_confdiff_loss, pair_confdiff_selected_ratio = self.get_pair_confdiff_ranking_loss(
            ecp_logits,
            ecp_mask,
            gold_matrix,
            emotion_logits,
            cause_logits,
        )
        pair_confdiff_weight = self.get_pair_confdiff_loss_weight()
        robust_pair_loss = ecp_logits.new_zeros(())
        robust_pair_consistency = ecp_logits.new_zeros(())
        corruption_ratio = 0.0
        if self.training and self.pair_robust_enable:
            corruption_mask = self.build_pair_robust_mask(
                utterance_nums, pairs, pair_nums, multimodal_output_pair.shape[1], multimodal_output_pair.device
            )
            if corruption_mask.any():
                corruption_ratio = float(corruption_mask.float().mean().detach().item())
                corrupted_text = self.apply_pair_robust_corruption(text_encoded, corruption_mask)
                corrupted_audio = self.apply_pair_robust_corruption(audio_encoded, corruption_mask)
                corrupted_video = self.apply_pair_robust_corruption(video_encoded, corruption_mask)
                corrupted_pair_input = self.apply_pair_robust_corruption(text_with_speaker, corruption_mask)
                corrupted_output_pair = self.process_multimodal_kerple(
                    corrupted_text,
                    corrupted_audio,
                    corrupted_video,
                    corrupted_pair_input,
                    utterance_nums,
                    utdg_seq_params,
                )
                robust_pair_loss, robust_pair_logits = self.get_dot_product(
                    corrupted_output_pair, ecp_mask, gold_matrix, similarity_matrix
                )
                robust_pair_consistency = self.get_pair_robust_consistency_loss(
                    ecp_logits, robust_pair_logits, ecp_mask
                )
        teacher_pair_loss = self.get_teacher_pair_ranking_loss(
            ecp_logits,
            ecp_mask,
            teacher_pair_positive_mask=teacher_pair_positive_mask,
            teacher_pair_negative_mask=teacher_pair_negative_mask,
        )
        teacher_loss_weight = self.get_teacher_loss_weight()

        pair_loss_weight = float(getattr(self.cfg, 'pair_role_loss_weight', 1.0)) if use_pair_role else 1.0
        context_subtype_loss_weight = float(getattr(self.cfg, 'context_subtype_loss_weight', 1.0))
        emotion_loss_weight = float(getattr(self.cfg, 'emotion_loss_weight', 1.0))
        cause_loss_weight = float(getattr(self.cfg, 'cause_loss_weight', 1.0))
        loss = (
            (emotion_loss_weight * emo_loss)
            + (cause_loss_weight * cause_loss)
            + (pair_loss_weight * ecp_loss)
            + (teacher_loss_weight * teacher_pair_loss)
            + (context_subtype_loss_weight * context_subtype_loss)
            + (pair_confdiff_weight * pair_confdiff_loss)
            + (self.pair_robust_corrupt_loss_weight * robust_pair_loss)
            + (self.pair_robust_consistency_weight * robust_pair_consistency)
        )
        final_logits = ecp_logits
        self.latest_loss_components = {
            'emo_loss': float(emo_loss.detach().item()),
            'emotion_loss_weight': float(emotion_loss_weight),
            'cause_loss': float(cause_loss.detach().item()),
            'cause_loss_weight': float(cause_loss_weight),
            'pair_loss': float(ecp_loss.detach().item()),
            'pair_loss_weight': float(pair_loss_weight),
            'pair_role_main_loss': float(getattr(self, '_last_pair_role_main_loss', ecp_loss.detach()).item()),
            'pair_role_related_loss': float(getattr(self, '_last_pair_role_related_loss', ecp_loss.detach().new_zeros(())).item()),
            'pair_role_related_loss_weight': float(getattr(self.cfg, 'pair_role_related_loss_weight', 0.0)),
            'context_subtype_loss': float(context_subtype_loss.detach().item()),
            'context_subtype_loss_weight': float(context_subtype_loss_weight),
            'pair_confdiff_loss': float(pair_confdiff_loss.detach().item()),
            'pair_confdiff_weight': float(pair_confdiff_weight),
            'pair_confdiff_selected_ratio': float(pair_confdiff_selected_ratio),
            'pair_robust_corrupt_loss': float(robust_pair_loss.detach().item()),
            'pair_robust_consistency': float(robust_pair_consistency.detach().item()),
            'pair_robust_corrupt_weight': float(self.pair_robust_corrupt_loss_weight),
            'pair_robust_consistency_weight': float(self.pair_robust_consistency_weight),
            'pair_robust_corruption_ratio': float(corruption_ratio),
            'teacher_pair_rank': float(teacher_pair_loss.detach().item()),
            'teacher_loss_weight': float(teacher_loss_weight),
        }

        return loss, (final_logits, emotion_logits, cause_logits, ecp_mask, context_subtype_logits)


TextClassification = PaceExtModel
