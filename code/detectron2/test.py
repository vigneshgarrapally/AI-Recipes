import argparse
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from pathlib import Path,PurePosixPath
import config
from detectron2.utils.logger import setup_logger
import boto3
import os
from dotenv import load_dotenv
import logging
for name in logging.root.manager.loggerDict:
    if not "botocore" in name and not "boto3" in name:
        logging.getLogger(name).propagate=True

log_output_dir = Path("../outputs/trail {}/logs/test.log".format(config.TEST_TRAIL))
log_output_dir.parent.mkdir(exist_ok=True,parents=True)
logger=setup_logger(name="test", output=str(log_output_dir))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(str(log_output_dir))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
def build_args():
    parser=argparse.ArgumentParser(description='Evaluate the model',usage='%(prog)s [options] path',)
    parser.add_argument('--dataset',type=str,help='Path of Dataset to test. If not given, look for the test dataset in the config.py file')
    parser.add_argument('--test_trail',type=int,help='which trail to be tested')
    return parser.parse_args()

def main(args):
    if args.test_trail is None:
        args.test_trail=config.TEST_TRAIL
    if args.dataset is None:
        args.dataset=config.TEST_PATH
    output_path=Path("../outputs/")/"trail {}".format(args.test_trail)
    if not(output_path.exists() and Path(args.dataset).exists()):
        logger.error("Either config file or dataset path is not found")
        raise Exception("Either config file or dataset path is not found")
    cfg = get_cfg()
    cfg.merge_from_file(str(output_path/"output_config.yaml"))
    register_coco_instances("test",{},Path(args.dataset)/"annotations/instances_default.json",Path(args.dataset)/"images")
    cfg.DATASETS.TEST = ("test",)
    model_path=Path(cfg.MODEL.WEIGHTS)
    if not model_path.exists():
        logger.info("Downloading the model from s3")
        print("Downloading the model from s3")
        load_dotenv("../../.env")
        s3_client=boto3.client('s3',aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                            aws_secret_access_key=os.environ['AWS_SECRET_KEY'], 
                            region_name=os.environ['AWS_DEFAULT_REGION'])
        s3_client.download_file(os.environ['S3_BUCKET_NAME'],os.ENVIRON['AWS_FOLDER_NAME']+PurePosixPath(model_path.relative_to("../outputs/")).__str__(),str(model_path))
        logger.info("Model downloaded")
    logger.info("Loading the model")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "test")
    logger.info(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    args=build_args()
    main(args)
