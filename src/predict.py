#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Modified from this example:
    https://github.com/fizyr/keras-retinanet/blob/master/examples/ResNet50RetinaNet.ipynb
"""

import os, os.path
import time
import csv
import argparse
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='Jelly Bean Finder.')
    parser.add_argument("model", type=str, help="File for indication model.")
    parser.add_argument("out_dir", type=str, help="Output directory for labeled prediction.")
    parser.add_argument("label", type=str, help="CSV file for annotation labels.")
    parser.add_argument("image", type=str, nargs="+", help="Image files to be labeled.")
    parser.add_argument("--backbone", required=False, type=str, default="resnet50", \
                        dest="backbone", help="Type of backbone model.")
    parser.add_argument("--outprefix", required=False, type=str, default="result", \
                        dest="out_prefix", help="Prefix for output files.")
    return parser.parse_args()

def test_args(args):
    if not os.path.isfile(args.model):
        print("{:s} is not a valid file.".format(args.model))
        return False
    if not os.path.isdir(args.out_dir):
        print("{:s} is not a valid directory.".format(args.out_dir))
        return False
    if not os.path.isfile(args.label):
        print("{:s} is not a valid file.".format(args.label))
    if not all(map(os.path.isfile, args.image)):
        print("Some of the image files are not valid.")
        return False
    return True

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def get_labels(f):
    result = {}
    with open(f, "r", newline='') as fd_r:
        rdr = csv.DictReader(fd_r, fieldnames=["NAME", "ID"])
        for row in rdr:
            result[int(row["ID"])] = row["NAME"]
    return result

def run():
    args = parse_args()
    print(args)
    if not test_args(args):
        return
    keras.backend.tensorflow_backend.set_session(get_session())
    label_map = get_labels(args.label)
    model = models.load_model(args.model, backbone_name=args.backbone)

    for img_file in args.image:
        img = read_image_bgr(img_file)
        canvas = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img, scale = resize_image(preprocess_image(img))
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(img, axis=0))
        print("Processing time: ", time.time() - start)

        boxes /= scale
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.45: # Labels are sorted
                 break
            color = label_color(label)
            b = box.astype(int)
            draw_box(canvas, b, color=color)
            caption = "{} {:.3f}".format(label_map[label], score)
            draw_caption(canvas, b, caption)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        out_file = args.out_prefix + os.path.split(img_file)[1]
        out_full = os.path.join(args.out_dir, out_file)
        cv2.imwrite(out_full, canvas)
        print("Done with writing to file {:s}.".format(out_full))

if __name__ == "__main__":
    run()
