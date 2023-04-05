import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import time
import cv2
import tqdm
import config
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from pathlib import Path,PurePosixPath
from predictor import VisualizationDemo
import os
from dotenv import load_dotenv
# constants
WINDOW_NAME = "Inference"
import logging

log_output_dir = Path("../outputs/trail {}/logs/infer.log".format(config.INFERENCE_TRAIL_NO))
log_output_dir.parent.mkdir(exist_ok=True,parents=True)
logger=setup_logger(name="infer", output=str(log_output_dir))

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    if args.config_file is None:
        args.config_file=config.CONFIG_FILE
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold if args.confidence_threshold is not None else config.INFERENCE_THRESHOLD
    cfg.freeze()
    model_path=Path(cfg.MODEL.WEIGHTS)
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file. If None, looks for config path in config.py",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    if args.config_file is None:
        args.config_file = config.INFERENCE_CONFIG
    if args.input is None:
        args.input = config.INFERENCE_IMAGE_PATHS
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            for i in range(4):
                if i==1:
                    start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    (time.time() - start_time)/4,
                )
            )
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                cv2.imwrite("output.jpg",visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit