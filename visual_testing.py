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


def main(config: Dict[str, Any], args: argparse.Namespace):
    device = torch.device("cuda") if tcuda.is_available() else torch.device("cpu")
    model = IDisc.build(config)
    model.load_pretrained(args.model_file)
    model = model.to(device)
    model.eval()

    f16 = config["training"].get("f16", False)
    context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=f16)

    img = Image.open(
        "I:\\Research-on-MDE-in-mining-scenarios\\kitti\\raw\\2011_09_29\\2011_09_29_drive_0004_sync\\image_02\\data\\0000000277.png")
    transform = trasforms.Compose([trasforms.ToTensor()])
    img = transform(img).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        preds, losses, _ = model(img.to(device), None, None)
    img = visulization.colorize(preds.cpu().numpy())
    img = Image.fromarray(img.astype('uint8'))
    img.save("output.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing", conflict_handler="resolve")
    parser.add_argument("--config-file", type=str, default="configs/kitti/kitti_r101.json")
    parser.add_argument("--model-file", type=str, default="pretrained/kitti_resnet101.pt")
    parser.add_argument("--base-path", default=".")

    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = json.load(f)

    main(config, args)
