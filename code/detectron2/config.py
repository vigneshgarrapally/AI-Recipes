#PYTHON FILE FOR DETECTRON2 CONFIGURATION

# PARAMS FOR train.py

TRAIL_NO=1

TRAIN_PATH='../dataset-2/train/'

TEST_PATH='../dataset-2/test/'

CONFIG_FILE='mask_rcnn_R_50_FPN_3x.yaml'

UPLOAD= False

#PARAMS FOR test.py

TEST_TRAIL=1

TEST_PATH='../dataset-2/test/'

DOWNLOAD=False

# PARAMS FOR infer.py
INFERENCE_THRESHOLD=0.5

INFERENCE_TRAIL_NO=1

INFERENCE_IMAGE_PATHS=["D:\\Yottaasys\\assa_abloy\\Assa-Abloy\\detectron2\\dataset-2\\test\\images\\test_frames_SKI\\8\\os.png"]

INFERENCE_CONFIG='../outputs/trail {}/output_config.yaml'.format(INFERENCE_TRAIL_NO)

DOWNLOAD=False
