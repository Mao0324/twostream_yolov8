import argparse
import torch
import torch.nn as nn

#python tools/check_cpt.py --weights 路径/weights/best.pt

def pick_state_dict(ckpt):
    # 1) 直接是 nn.Module
    if isinstance(ckpt, nn.Module):
        return ckpt.float().state_dict()

    # 2) 常见 Ultralytics checkpoint dict
    if isinstance(ckpt, dict):
        # 优先 ema，其次 model（两者可能有一个是 None）
        for k in ("ema", "model"):
            v = ckpt.get(k, None)
            if isinstance(v, nn.Module):
                return v.float().state_dict()

        # 有些 checkpoint 只有 state_dict
        if isinstance(ckpt.get("state_dict", None), dict):
            return ckpt["state_dict"]

        # 也可能本身就是参数字典
        if len(ckpt) and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

        raise RuntimeError(
            f"checkpoint 里没有可用模型。keys={list(ckpt.keys())}, "
            f"model={type(ckpt.get('model'))}, ema={type(ckpt.get('ema'))}"
        )

    raise RuntimeError(f"不支持的 checkpoint 类型: {type(ckpt)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="pt 权重路径")
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    sd = pick_state_dict(ckpt)

    betas = {k: float(v) for k, v in sd.items() if k.endswith(".beta")}
    temps = {k: float(v) for k, v in sd.items() if k.endswith(".temperature")}

    print("beta:")
    for k, v in betas.items():
        print(f"  {k}: {v:.6f}")

    print("temperature:")
    for k, v in temps.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
