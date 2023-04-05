#import required modules
import argparse
import config
import shutil
import os
import boto3
from dotenv import load_dotenv
from pathlib import Path
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
import logging


#Instantiate the logger with DEBUG mode. logs are to mentioned address
log_output_dir = Path("../outputs/trail {}/logs/train.log".format(config.TRAIL_NO))
log_output_dir.parent.mkdir(exist_ok=True,parents=True)
logger=setup_logger(name="train", output=str(log_output_dir))
fh = logging.FileHandler(str(log_output_dir))
fh.setLevel(logging.DEBUG)
for name in logging.root.manager.loggerDict:
    if not "botocore" in name and not "boto3" in name:
        logging.getLogger(name).addHandler(fh)

def main(args):
    if args.config is None:
        args.config=config.CONFIG_FILE
    if args.train_data is None:
        args.train_data=config.TRAIN_PATH
    if args.test_data is None:
        args.test_data=config.TEST_PATH
    print(args)
    if not(Path(args.config).exists() and Path(args.train_data).exists() and Path(args.test_data).exists()):
        logger.error("Either config file or dataset path is not found")
        raise Exception("Either config file or dataset path is not found")
    register_coco_instances("train",{},Path(args.train_data)/"annotations/instances_default.json",Path(args.train_data)/"images")
    register_coco_instances("test",{},Path(args.test_data)/"annotations/instances_default.json",Path(args.test_data)/"images")
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("test",)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 600
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.OUTPUT_DIR = "../outputs/trail {}".format(config.TRAIL_NO)
    logger.info("Dataset is registered")
    output_folder=Path("../outputs/")/"trail {}".format(config.TRAIL_NO)
    shutil.rmtree(output_folder,ignore_errors=True)
    output_folder.mkdir(exist_ok=True,parents=True)
    print(DatasetCatalog.get("train")[0])
    cfg.OUTPUT_DIR=str(output_folder)
    trainer=DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = str(Path(cfg.OUTPUT_DIR)/"model_final.pth")
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("test", output_dir=str(output_folder))
    val_loader = build_detection_test_loader(cfg, "test")
    logger.info(inference_on_dataset(predictor.model, val_loader, evaluator))
    #save the config file
    with open(Path(output_folder)/"output_config.yaml","w") as f:
        f.write(cfg.dump())
        logger.info("Config file saved at {}".format(Path(output_folder)/"output_config.yaml"))
        #upload final model to s3
    if config.UPLOAD:
        logger.info("Uploading the model to s3")
        print("Uploading the model to s3")
        load_dotenv("../../.env")
        s3_client=boto3.client('s3',aws_access_key_id=os.environ['AWS_ACCESS_KEY'], 
                            aws_secret_access_key=os.environ['AWS_SECRET_KEY'], 
                            region_name=os.environ['AWS_DEFAULT_REGION'])
        s3_client.upload_file(str(Path(output_folder)/"model_final.pth"),os.environ['AWS_BUCKET_NAME'],os.environ['AWS_FOLDER_NAME']+"trail {}/model_final.pth".format(config.TRAIL_NO))
        logger.info("Model uploaded to {}".format(os.environ['AWS_FOLDER_NAME']+"model_final.pth"))


def build_args():
    parser=argparse.ArgumentParser(description='Train the model',usage='%(prog)s [options] path',)
    parser.add_argument('--train-data',type=str,help='Path of Dataset to train. If not given, look for the dataset in the config.py file')
    parser.add_argument('--trail-no',type=int,help='Trail number. If not given, look for the trail number in the config.py file')
    parser.add_argument('--upload',type=bool,help='Upload the model to s3. If not given, look for the upload flag in the config.py file')
    parser.add_argument('--test-data',type=str,help='Path of Dataset to test. If not given, look for the dataset in the config.py file')
    parser.add_argument('--config','-c',type=str,help='Path of config file. If not given, look for the config file in the config.py file')
    return parser.parse_args()

if __name__=='__main__':
    args=build_args()
    main(args)
    logger.info("Training completed")

    
