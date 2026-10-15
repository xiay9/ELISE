
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import warnings
import weakref

class EnhancedLSTM(torch.nn.Module):
    """LSTM with drop-connect recurrent dropout."""

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.provider = WeightDropLSTM(
            input_size,
            hidden_size,
            num_layers,
            ff_dropout,
            recurrent_dropout,
            bidirectional=bidirectional)

    # Expects unpacked inputs in format (batch, seq, features)
    def forward(self, inputs, hidden, lengths):
        seq_len = inputs.shape[1]
        return self.provider(inputs, lengths, seq_len)


class WeightDropLSTM(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 ff_dropout: float = 0.0,
                 recurrent_dropout: float = 0.0,
                 bidirectional=True) -> None:
        super().__init__()

        self.locked_dropout = LockedDropout()
        self.lstms = [
            torch.nn.LSTM(
                input_size
                if l == 0 else hidden_size * (1 + int(bidirectional)),
                hidden_size,
                num_layers=1,
                dropout=0,
                bidirectional=bidirectional,
                batch_first=True) for l in range(num_layers)
        ]
        if recurrent_dropout:
            self.lstms = [
                WeightDrop(lstm, ['weight_hh_l0'], dropout=recurrent_dropout)
                for lstm in self.lstms
            ]

        self.lstms = torch.nn.ModuleList(self.lstms)
        self.ff_dropout = ff_dropout
        self.num_layers = num_layers

    def forward(self, input, lengths, seq_len):
        """Expects input in format (batch, seq, features)"""
        output = input
        for lstm in self.lstms:
            output = self.locked_dropout(
                output, batch_first=True, p=self.ff_dropout)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                output, lengths, batch_first=True, enforce_sorted=False)
            output, _ = lstm(packed, None)
            output, _ = torch.nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=seq_len)

        return output

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch_first=False, p=0.5):
        if not self.training or not p:
            return x
        mask_shape = (x.size(0), 1, x.size(2)) if batch_first else (1,
                                                                    x.size(1),
                                                                    x.size(2))

        mask = x.data.new(*mask_shape).bernoulli_(1 - p).div_(1 - p)
        return mask * x



class WeightDrop(torch.nn.Module):
    def __init__(self, module, weights, dropout=0):
        super(WeightDrop, self).__init__()
        self.module = module
        self.weights = weights
        if hasattr(module, "bidirectional") and module.bidirectional:
            self.weights.extend(
                [weight + "_reverse" for weight in self.weights])

        self.dropout = dropout
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            if name_w in self.module._parameters:
                del self.module._parameters[name_w]
            self.register_parameter(name_w + '_raw', Parameter(w.data))

    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self, name_w + '_raw')

            w = None
            mask = torch.ones(1, raw_w.size(1))
            if raw_w.is_cuda: mask = mask.to(raw_w.device)
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout, training=self.training)
            w = mask.expand_as(raw_w) * raw_w
            setattr(self.module, name_w, w)

        # torch>=2.8: RNNBase._apply may have built _flat_weights with None entries.
        # Keep the LSTM's internal flat weight cache in sync to avoid passing None to _VF.lstm.
        if hasattr(self.module, "_flat_weights_names"):
            self.module._flat_weights = [
                getattr(self.module, wn) if hasattr(self.module, wn) else None
                for wn in self.module._flat_weights_names
            ]
            self.module._flat_weight_refs = [
                weakref.ref(w) if w is not None else None
                for w in self.module._flat_weights
            ]

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # Ignore lack of flattening warning
            warnings.simplefilter("ignore")
            return self.module.forward(*args)


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1) / self.n_in ** self.scale

        return s


class KERPLEDecayKernel(nn.Module):
    """
    KERPLE-Lite exponential decay kernel.
    """
    def __init__(self, max_len=50, num_modalities=3):
        super(KERPLEDecayKernel, self).__init__()
        self.max_len = max_len
        self.num_modalities = num_modalities

        # λ_m: modality-specific decay rates [text, audio, video]
        self.decay_rates = nn.Parameter(torch.tensor([0.1, 0.3, 0.05], dtype=torch.float32))

    def compute_kernel_weights_batch(self, seq_len, lengths, modality_id=0, device='cpu', param_override=None):
        """
        批量计算KERPLE-Lite核权重矩阵
        Args:
            seq_len: 最大序列长度
            lengths: [batch_size] 实际长度
            modality_id: 模态ID
            device: 设备
            param_override: [batch_size] 可选参数覆盖
        Returns:
            kernel_matrix: [batch_size, seq_len, seq_len]
        """
        batch_size = lengths.size(0)
        distances = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        distances = torch.abs(distances).float()

        decay_rate = self.decay_rates[modality_id]
        if param_override is not None:
            decay_rate = torch.as_tensor(param_override, device=device, dtype=torch.float32)
        if decay_rate.dim() == 0:
            decay_rate = decay_rate.view(1)
        decay_rate = decay_rate.view(-1, 1, 1)
        weights = torch.exp(-decay_rate * distances)

        # 下三角掩码（因果性约束）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        weights = weights * mask

        valid = torch.arange(seq_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        weights = weights * valid.to(weights.dtype)

        row_sums = weights.sum(dim=2, keepdim=True)
        weights = weights / (row_sums + 1e-8)
        return weights


class KERPLEEnhancedLSTM(nn.Module):
    """
    KERPLE-Lite适配的增强LSTM
    核心思想：在隐藏状态更新时加入核衰减的历史加权
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 max_len=50,
                 ff_dropout=0.1, recurrent_dropout=0.1, bidirectional=True):
        super(KERPLEEnhancedLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # KERPLE核衰减模块
        self.decay_kernel = KERPLEDecayKernel(max_len=max_len)

        # 基础LSTM层
        self.base_lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=ff_dropout if num_layers > 1 else 0
        )

        # 历史状态记忆融合层
        self.memory_fusion = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions * 2, hidden_size * self.num_directions),
            nn.Tanh(),
            nn.Dropout(recurrent_dropout)
        )

        # 多模态适配参数
        self.modality_weights = nn.Parameter(torch.ones(3))  # [text, audio, video]权重

    def _get_param_override_batch(self, utdp_params, batch_size, modality_id, device):
        if utdp_params is None:
            return None
        if not torch.is_tensor(utdp_params):
            raise TypeError(f"utdp_params must be a torch.Tensor, got {type(utdp_params)}")
        if utdp_params.dim() >= 2:
            if batch_size > utdp_params.size(0) or modality_id >= utdp_params.size(1):
                raise IndexError(
                    f"utdp_params shape={tuple(utdp_params.shape)} but batch_size={batch_size}, modality_id={modality_id}"
                )
            param_override = utdp_params[:batch_size, modality_id]
        elif utdp_params.dim() == 1:
            if modality_id >= utdp_params.size(0):
                raise IndexError(f"utdp_params shape={tuple(utdp_params.shape)} but modality_id={modality_id}")
            param_override = utdp_params[modality_id].expand(batch_size)
        else:
            raise ValueError(f"utdp_params has invalid dim={utdp_params.dim()}")

        if param_override.dim() != 1:
            param_override = param_override.reshape(batch_size, -1)
            if param_override.size(1) != 1:
                raise ValueError(
                    f"utdp_params should provide scalar per batch, got shape={tuple(param_override.shape)}"
                )
            param_override = param_override.squeeze(1)
        return param_override.to(device)

    def forward(self, input_seq, lengths, modality_id=0, utdp_params=None):
        """
        KERPLE-Lite增强的LSTM前向传播
        Args:
            input_seq: [batch_size, seq_len, input_size]
            lengths: [batch_size] 实际序列长度
            modality_id: 当前处理的模态ID
        Returns:
            enhanced_output: [batch_size, seq_len, hidden_size * num_directions]
        """
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        # 1. 基础LSTM前向传播
        packed_input = nn.utils.rnn.pack_padded_sequence(
            input_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.base_lstm(packed_input)
        base_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )

        lengths_device = lengths.to(device)

        param_override = None
        if utdp_params is not None:
            param_override = self._get_param_override_batch(utdp_params, batch_size, modality_id, device)

        kernel_weights = self.decay_kernel.compute_kernel_weights_batch(
            seq_len, lengths_device, modality_id, device, param_override=param_override
        )  # [batch_size, seq_len, seq_len]

        kerple_output = torch.bmm(kernel_weights, base_output)
        combined_features = torch.cat([base_output, kerple_output], dim=-1)
        fused_output = self.memory_fusion(combined_features)

        modality_weight = torch.softmax(self.modality_weights, dim=0)[modality_id]
        final_output = modality_weight * fused_output + (1 - modality_weight) * base_output

        valid_mask = torch.arange(seq_len, device=device).unsqueeze(0) < lengths_device.unsqueeze(1)
        final_output = final_output * valid_mask.unsqueeze(-1)
        return final_output
