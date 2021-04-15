import cv2
import glob
import re
import numpy as np
import os
import argparse
import json

from pathlib import Path

#example
#python image_crop.py --data_dir /home/zhoulin/dataset/front --boxs [[41, 143, 63, 60], [158, 142, 63, 60]] --save_dir .results --partrtn *1.bmp

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/home/zhoulin/dataset')
parser.add_argument('--boxs', type=json.loads, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--inherit_dir_layers', '-i', type=int, default=2)
parser.add_argument('--partern', type=str, default='*.*')


opt = parser.parse_args()
opt.data_dir = Path(opt.data_dir)
if not opt.save_dir:
    opt.save_dir = Path('.results')
else:
    opt.save_dir = Path(opt.save_dir)

if not opt.save_dir.exists():
    opt.save_dir.mkdir(exist_ok=True, parents=True)


def crop(img_path, boxs, outdir):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    for box in boxs:
        x0 = box[0]
        y0 = box[1]
        x1 = box[0] + box[2]
        y1 = box[1] + box[3]
        cropped = img[y0:y1, x0:x1]
        img_name = img_path.stem + '_box' + str(boxs.index(box)) + img_path.suffix
        i = 0
        while i < opt.inherit_dir_layers:
            img_name = img_path.parents[i].name + '_' + img_name
            i = i + 1
        cv2.imwrite("{0}/{1}".format(outdir, img_name), cropped) #裁剪并存储在指定文件夹中


if __name__ == "__main__":

    if not opt.data_dir.exists():
        print("Please check the data dir.")
        exit
    for f in opt.data_dir.rglob(opt.partern):
        crop(f, opt.boxs, opt.save_dir)
