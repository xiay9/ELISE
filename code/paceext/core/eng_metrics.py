#!/usr/bin/env python

import time
import numpy as np
import torch


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': size_mb
    }


def collect_eng_metrics(model, data_loader, device, warmup_full=True):
    model.eval()
    latencies_ms = []
    flops_totals = []
    total_samples = 0
    total_batches = 0

    if warmup_full:
        for data in data_loader:
            data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            with torch.no_grad():
                _ = model(**data)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    try:
        from torch.profiler import profile, ProfilerActivity
        use_profiler = True
    except Exception:
        use_profiler = False

    activities = [ProfilerActivity.CPU] if use_profiler else []
    if use_profiler and device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    for data in data_loader:
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
        batch_size = int(data['input_ids'].size(0)) if 'input_ids' in data else 1
        total_samples += batch_size
        total_batches += 1

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if use_profiler:
            with torch.no_grad():
                with profile(activities=activities, with_flops=True, profile_memory=False, record_shapes=False) as prof:
                    _ = model(**data)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            total_flops = 0
            for item in prof.key_averages():
                flops = getattr(item, 'flops', None)
                if flops:
                    total_flops += flops
            flops_totals.append(float(total_flops))
        else:
            with torch.no_grad():
                _ = model(**data)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms / max(batch_size, 1))

    latency_stats = {}
    if latencies_ms:
        p50 = float(np.percentile(latencies_ms, 50))
        p95 = float(np.percentile(latencies_ms, 95))
        avg = float(np.mean(latencies_ms))
        throughput = float(1000.0 / avg) if avg > 0 else 0.0
        latency_stats = {
            'latency_p50_ms': p50,
            'latency_p95_ms': p95,
            'latency_avg_ms': avg,
            'latency_p95_over_p50': float(p95 / p50) if p50 > 0 else 0.0,
            'throughput_samples_per_s': throughput
        }

    flops_stats = {}
    if flops_totals:
        total_flops = float(np.sum(flops_totals))
        avg_flops_per_batch = total_flops / max(total_batches, 1)
        per_sample = total_flops / max(total_samples, 1)
        flops_stats = {
            'flops_per_batch': avg_flops_per_batch,
            'flops_per_sample': per_sample,
            'gflops_per_sample': per_sample / 1e9
        }

    return latency_stats, flops_stats
