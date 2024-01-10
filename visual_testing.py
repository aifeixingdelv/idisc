import argparse
import json
import os
from typing import Any, Dict
import numpy as np
import torch
import torch.cuda as tcuda
import torch.utils.data.distributed
from torch.utils.data import DataLoader, SequentialSampler
import idisc.dataloders as custom_dataset
from idisc.models.idisc import IDisc
from PIL import Image
from idisc.utils import (DICT_METRICS_DEPTH, DICT_METRICS_NORMALS,
                         RunningMetric, validate)
import torchvision.transforms as trasforms
import idisc.utils.visulization as visulization
from matplotlib import pyplot as plt
import torchvision.transforms.functional as TF


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    img = Image.open(
        "datasets\\AutoMine-Depth\\OCT-00\\scene_01\\raw\\1661922775.200000.png")
    # img = img.crop((200, 100, 1100, 350))
    img = TF.normalize(TF.to_tensor(img),  **{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
    img = img.unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        preds, *_ = model(img.to(device))
    preds = preds.squeeze().cpu().numpy()
    img = visulization.colorize(preds, vmin=0.01, vmax=80, cmap="magma_r")
    img = Image.fromarray(img)

    min_val = 0.01
    max_val = 80
    scaled_array = (preds - min_val) / (max_val - min_val)
    preds = scaled_array * 255
    preds = Image.fromarray(preds.astype("uint8"), mode='L')
    preds.save("gray.png")
    img.save("RGB.png")
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")
    parser.add_argument("--config-file", type=str, default="configs/automine/automine_r101.json")
    parser.add_argument("--model-file", type=str, default="pretrained/kitti_resnet101.pt")
    parser.add_argument("--base-path", default=".")

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config, args)
