from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from pathlib import Path
import random
import cv2
import argparse
import config

def build_args():
    parser=argparse.ArgumentParser(description='Verify Annotations of the dataset',usage='%(prog)s [options] path',)
    parser.add_argument('--number','-num',type=int,default=5,help='Number of images to visualize. Default is 5.  -1 for whole dataset')
    parser.add_argument('--dataset',type=str,help='Path of Dataset to visualize. If not given, look for the dataset in the config.py file')
    return parser.parse_args()

def visualize(name,dataset,number):
    dpath=Path(dataset)
    if not dpath.exists():
        raise Exception("Dataset path ({})does not exist. Please check the path".format(dpath))
    register_coco_instances(name,{},dpath/'annotations/instances_default.json',dpath/'images')
    dataset_dicts = DatasetCatalog.get(name)
    if number==-1:
        number=len(dataset_dicts)
    for i,d in enumerate(random.sample(dataset_dicts, number)):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(name), scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("visualization",out.get_image()[:, :, ::-1])
        cv2.imwrite("{}.jpg".format(i),out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
def main(args):
    #registering the dataset
    if args.dataset:
        visualize("vis_dataset",args.dataset,args.number)
    else:
        visualize("train",config.TRAIN_PATH,args.number)
        visualize("test",config.TEST_PATH,args.number)

if __name__ == "__main__":
    args=build_args()
    main(args)


