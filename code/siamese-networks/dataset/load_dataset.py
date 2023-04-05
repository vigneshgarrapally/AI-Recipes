# import the necessary packages
import tensorflow as tf
from dataset.utils import list_images
import config
from dataset.build_siamese_pairs import make_random_pairs
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
channels=config.channels
shape=config.IMAGE_SHAPE
def load_images(input1,input2,input3):
    imageA = tf.io.read_file(input1)
    imageA = tf.image.decode_image(imageA, channels=3,expand_animations=False)
    imageA = tf.image.resize(imageA, shape)
    imageA = tf.image.convert_image_dtype(imageA, tf.float32)
    imageB = tf.io.read_file(input2)
    imageB = tf.image.decode_image(imageB, channels=channels,expand_animations=False)
    imageB = tf.image.resize(imageB,shape)
    imageB = tf.image.convert_image_dtype(imageB, tf.float32)
    encodedLabel = input3
    # return the image and the integer encoded label
    return (imageA,imageB,encodedLabel)


trainAug = Sequential([
    preprocessing.RandomFlip("horizontal_and_vertical"),
    preprocessing.RandomZoom(
        height_factor=(-0.05, -0.15),
        width_factor=(-0.05, -0.15)),
    preprocessing.RandomRotation(0.3)
])


def build_dataset():
    #prepare train dataset
    train_paths=list_images(config.TRAIN_PATH)
    trainlabels = [f.parent.name for f in train_paths]
    train_paths=list(map(str, train_paths))
    assert len(train_paths) == len(trainlabels)
    #prepare val dataset
    val_paths=list_images(config.VAL_PATH)
    vallabels = [f.parent.name for f in val_paths]
    val_paths=list(map(str, val_paths))
    assert len(val_paths) == len(vallabels)
    #prepare test dataset
    test_paths=list_images(config.TEST_PATH)
    testlabels = [f.parent.name for f in test_paths]
    test_paths=list(map(str, test_paths))
    assert len(test_paths) == len(testlabels)


    trainA,trainB,trainY,dictmap=make_random_pairs(train_paths,trainlabels,labelsmap=True)
    valA,valB,valY,_=make_random_pairs(val_paths,vallabels,labelsmap=True)
    testA,testB,testY,_=make_random_pairs(test_paths,testlabels,labelsmap=True)


    print("[INFO] creating a tf.data input pipeline..")
    tA=tf.data.Dataset.from_tensor_slices(trainA)
    tB=tf.data.Dataset.from_tensor_slices(trainB)
    tY=tf.data.Dataset.from_tensor_slices(trainY)
    trainDS = tf.data.Dataset.zip((tA,tB,tY))
    trainDS = (trainDS
        .shuffle(len(trainA))
        .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x,y,z: ((trainAug(x),trainAug(y)),z))
        .cache()
        .batch(config.BS)
        .prefetch(tf.data.AUTOTUNE)
    )

    vA=tf.data.Dataset.from_tensor_slices(valA)
    vB=tf.data.Dataset.from_tensor_slices(valB)
    vY=tf.data.Dataset.from_tensor_slices(valY)
    valDS = tf.data.Dataset.zip((vA,vB,vY))
    valDS = (valDS
        .shuffle(len(valA))
        .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(config.BS)
        .prefetch(tf.data.AUTOTUNE)
    )

    #build test data pipeline
    tA=tf.data.Dataset.from_tensor_slices(testA)
    tB=tf.data.Dataset.from_tensor_slices(testB)
    tY=tf.data.Dataset.from_tensor_slices(testY)
    testDS = tf.data.Dataset.zip((tA,tB,tY))
    testDS = (testDS
        .map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(1)
        .prefetch(tf.data.AUTOTUNE)
    )
    return trainDS,valDS,testDS,dictmap