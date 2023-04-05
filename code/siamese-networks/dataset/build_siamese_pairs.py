# USAGE
# python build_siamese_pairs.py
from pathlib import Path
import numpy as np
import cv2
import argparse
from imutils import build_montages
#ideally we can generate nC2 pairs but using this strategy, we will generate n random pairs
def make_random_pairs(images, labels,labelsmap=False):
	if(isinstance(labels[0],str)):
		encoded=False
	elif (isinstance(int(labels[0]),int)):
		encoded=True
	else:
		raise ValueError("labels must be either int or str")
	# initialize two empty lists to hold the (image, image) pairs and labels
	ImagesA = []
	ImagesB = []
	pairLabels=[]
	numClasses = len(np.unique(labels))
	if not encoded:
		classnames = np.array(sorted(np.unique(labels)))
		labels=list(map(lambda x: np.argmax(x==classnames),labels))
		dictmap=dict(zip(list(range(len(classnames))),classnames))
	idx = [np.where(np.array(labels) == i)[0] for i in range(numClasses)]
	assert sum(list(map(len,idx)))==len(labels)
	for idxA in range(len(images)):
		# grab the current image and label belonging to the current iteration
		currentImage = images[idxA]
		label = labels[idxA]

		# grab a random image from same class
		idxB = np.random.choice(idx[label])
		posImage = images[idxB]

		#append a positive pair
		ImagesA.append(currentImage)
		ImagesB.append(posImage)
		pairLabels.append([1])

		# grab a random image from the different class label
		negIdx = np.random.choice(np.where(labels != label)[0])
		negImage = images[negIdx]

		# append a negative pair
		ImagesA.append(currentImage)
		ImagesB.append(negImage)
		pairLabels.append([0])
	
	if labelsmap:
		if encoded:
			raise Exception("Cannot return dict map for already encoded labels")
		# return a 2-tuple of our image pairs and labels
		return ImagesA,ImagesB,pairLabels,dictmap
	else:
		# return a 2-tuple of our image pairs and labels and a dict which maps the label to the classname
		return ImagesA,ImagesB,pairLabels


if __name__ == "__main__":
	from tensorflow.keras.datasets import mnist
	my_parser = argparse.ArgumentParser(description='Build Siamese Pairs')
	# Add the arguments
	my_parser.add_argument('--savepath',metavar='PATH',default="siamese_pairs.png",type=str,
                       						help='path to save image')
	my_parser.add_argument('--display',metavar='NUMBER',
					   default=[3,3],type=int,nargs=2,
                       help='Number of Images to build montages.')
	args = my_parser.parse_args()
	save_path=Path(args.savepath)
	if save_path.is_dir():
		save_path=Path(save_path, "siamese_pairs.png")
	rows=args.display[0]
	columns=args.display[1]
	# load MNIST dataset and scale the pixel values to the range of [0, 1]
	print("[INFO] loading MNIST dataset...")
	(_, _), (testX, testY) = mnist.load_data()

	# build the positive and negative image pairs
	print("[INFO] preparing positive and negative pairs...")
	testA,testB,testY = make_random_pairs(testX, testY)

	images = []

	# loop over a sample of our training pairs
	for i in np.random.choice(np.arange(0, len(testA)), size=(49,)):
		# grab the current image pair and label
		imageA = testA[i]
		imageB = testB[i]
		label = testY[i]
		output = np.zeros((36, 60), dtype="uint8")
		pair = np.hstack([imageA, imageB])
		output[4:32, 0:56] = pair
		text = "neg" if label[0] == 0 else "pos"
		color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
		vis = cv2.merge([output] * 3)
		vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
		cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
			color, 2)
		# add the pair visualization to our list of output images
		images.append(vis)
	# construct the montage for the images
	montage = build_montages(images, (96, 51), (rows, columns))[0]
	cv2.imwrite("siamese_pairs.png", montage)