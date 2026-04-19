import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model

THIS_DIR = Path(__file__).resolve().parent
DEIT_DIR = THIS_DIR / "deit"

for path in (THIS_DIR, DEIT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from datasets import build_dataset
from engine import evaluate
import models  # noqa: F401
import models_v2  # noqa: F401
from pysimulation import configure_attention


def get_args_parser():
    parser = argparse.ArgumentParser("DeiT evaluation script")
    parser.add_argument("--model", default="deit_base_patch16_224", type=str, help="model name")
    parser.add_argument("--data-path", required=True, type=str, help="dataset path")
    parser.add_argument("--checkpoint", default="", help="optional checkpoint path or URL")
    parser.add_argument("--method", default=None, help="attention method (int_attention, idx_softmax_only, exaq_attention, quant_only)")
    parser.add_argument("--inp-quant-bit", type=int, default=8)
    parser.add_argument("--quant-bit", type=int, default=5)
    parser.add_argument("--zero-thr", type=float, default=6.6)
    parser.add_argument("--bitwidth", type=int, default=3, help="bitwidth for EXAQ")
    return parser


def infer_input_size(model_name):
    if model_name.endswith("_384"):
        return 384
    if model_name.endswith("_448"):
        return 448
    return 224


def load_checkpoint(path):
    if path.startswith("https"):
        return torch.hub.load_state_dict_from_url(path, map_location="cpu", check_hash=True)
    return torch.load(path, map_location="cpu")


def extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_model = extract_model_state(checkpoint)
    state_dict = model.state_dict()

    for key in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if key in checkpoint_model and key in state_dict and checkpoint_model[key].shape != state_dict[key].shape:
            print(f"Removing key {key} from checkpoint due to shape mismatch")
            del checkpoint_model[key]

    if (
        "pos_embed" in checkpoint_model
        and hasattr(model, "pos_embed")
        and hasattr(model, "patch_embed")
        and checkpoint_model["pos_embed"].shape != model.pos_embed.shape
    ):
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(msg)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.input_size = infer_input_size(args.model)
    args.data_set = "IMNET"
    args.eval_crop_ratio = 0.875
    num_workers = 10
    pin_mem = True
    batch_size = 64

    torch.manual_seed(0)
    np.random.seed(0)
    cudnn.benchmark = True

    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=torch.utils.data.SequentialSampler(dataset_val),
        batch_size=int(1.5 * batch_size),
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    configure_attention(
        method_name=args.method,
        inp_quant_bit=args.inp_quant_bit,
        quant_bit=args.quant_bit,
        zero_thr=args.zero_thr,
        bitwidth=args.bitwidth,
    )
    print(f"Using {args.method} for scaled dot product attention")

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=not args.checkpoint,
        num_classes=nb_classes,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
        img_size=args.input_size,
    )

    if args.checkpoint:
        load_model_checkpoint(model, args.checkpoint)

    model.to(device)
    test_stats = evaluate(data_loader_val, model, device)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
