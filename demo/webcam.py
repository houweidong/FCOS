# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2
import os

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.imports import import_file
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        '-c', '--cam',
        action='store_true',
        help='whether use the cam or val image list')
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    if args.cam:
        cam = cv2.VideoCapture(0)
        while True:
            start_time = time.time()
            ret_val, img = cam.read()
            composite = coco_demo.run_on_opencv_image(img)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow("COCO detections", composite)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    else:
        paths_catalog = import_file(
            "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
        )
        dataset_list = cfg.DATASETS.TEST
        for dataset_name in dataset_list:
            imspth = paths_catalog.DatasetCatalog.get(dataset_name)['args']['root']
        img_names = os.listdir(imspth)
        start_time = time.time()
        for name in img_names:
            img_path = os.path.join(imspth, name)
            try:
                img = cv2.imread(img_path)
            except Exception as info:
                print(info)
                continue

            composite = coco_demo.run_on_opencv_image(img)
            print("Time: {:.2f} s / img".format(time.time() - start_time))
            cv2.imshow('COCO detections', composite)
            cv2.waitKey(0)


if __name__ == "__main__":
    main()
