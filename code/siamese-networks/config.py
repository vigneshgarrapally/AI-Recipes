import os
from pathlib import Path

#Fill Paths here
TRAIN_PATH=Path()
TEST_PATH=Path()
VAL_PATH=Path()

#IMAGE_SHAPE
IMAGE_SHAPE=(224,224)
#BATCH_SIZE
BS=32

#Output Path
OUTPUT_PATH=Path(".","outputs")


channels=3
numClasses=None
#EPOCHS
EPOCHS=200
