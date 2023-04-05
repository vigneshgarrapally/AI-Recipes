#import necessary modules
from tensorflow.python.eager.context import num_gpus
from dataset import load_dataset as ld
from model import efficient_netv2 as efv2
import config
from visualization import plots
from pathlib import Path
from metrics import distances
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
#build dataset
print("[INFO] preparing positive and negative pairs...")
trainDS,valDS,testDS,dictmap=ld.build_dataset()

print("[INFO] building siamese network...")
imgA = Input(shape=config.IMAGE_SHAPE+(3,))
imgB = Input(shape=config.IMAGE_SHAPE+(3,))
featureExtractor = efv2.build_efv2smodel(config.IMG_SHAPE[0],config.IMG_SHAPE[1],num_classes=config.numClasses)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = Lambda(distances.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=(imgA, imgB), outputs=outputs)
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
H=model.fit(trainDS,epochs=10,validation_data=valDS)
plots.plot_metrics(H.history,Path(config.OUTPUT_PATH+"/metrics.png"))
model.save(Path(config.OUTPUT_PATH+"/model.h5"))
print(model.evaluate(testDS))