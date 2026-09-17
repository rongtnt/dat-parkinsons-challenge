"""int8 per-output-channel symmetric quantisation of a state_dict (weights >= 2-D); 1-D tensors kept fp16.
Pack: torch.save(quantize_sd(sd), path). Load: sd = dequantize_sd(torch.load(path, map_location='cpu'))."""
import torch

def quantize_sd(sd):
    out = {}
    for k, v in sd.items():
        v = v.detach().float().cpu() if v.is_floating_point() else v.detach().cpu()
        if v.is_floating_point() and v.ndim >= 2 and v.numel() > 4096:
            flat = v.reshape(v.shape[0], -1)
            scale = flat.abs().amax(1).clamp_min(1e-8) / 127.0
            q = torch.round(flat / scale[:, None]).clamp(-127, 127).to(torch.int8).reshape(v.shape)
            out[k] = {"q": q, "s": scale.half()}
        else:
            out[k] = v.half() if v.is_floating_point() else v
    return out

def dequantize_sd(sd):
    out = {}
    for k, v in sd.items():
        if isinstance(v, dict) and "q" in v:
            q, s = v["q"], v["s"].float()
            out[k] = (q.float().reshape(q.shape[0], -1) * s[:, None]).reshape(q.shape)
        else:
            out[k] = v.float() if torch.is_tensor(v) and v.is_floating_point() else v
    return out

if __name__ == "__main__":
    import sys, os
    src, dst = sys.argv[1], sys.argv[2]
    torch.save(quantize_sd(torch.load(src, map_location="cpu")), dst)
    print(os.path.getsize(src) // 1024, "KB ->", os.path.getsize(dst) // 1024, "KB")
