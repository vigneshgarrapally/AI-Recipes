#pip install opencv-python
import cv2
import numpy as np
import utils 
def build_montages(image_list, image_shape, montage_shape,single=False):
    """
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are displayed with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :param single: boolean, if True, random images are picked equal to montage_shape and builds single montage
    :return: list of montage images in numpy array format
    """
    if single:
        img_choices = np.random.choice(len(image_list), size=montage_shape[0]*montage_shape[1], replace=False)
        image_list=[image_list[i] for i in img_choices]
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    image_montages = []
    # create a montage with black pixels for empty space
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False
    for img in image_list:
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))
        start_new_img = False
        img = cv2.resize(img, image_shape)
        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                      dtype=np.uint8)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages

if __name__ == '__main__':
    from pathlib import Path
    import argparse
    myparser = argparse.ArgumentParser(description='Builds montages of images')
    myparser.add_argument('inputpath',type=str,help='path directory of images or single image file. If single,duplicates are created')
    myparser.add_argument('--savedir', default='.',type=str,help='Directory to store montages')
    # load single image
    myparser.add_argument('--single', action='store_true',default=True, help='if True, random images are picked equal to montage_shape and builds single montage')
    myparser.add_argument('--display', action='store_true',default=False, help='if True, images are displayed')
    myparser.add_argument('--image_shape', type=int, nargs=2,default=(256, 256), help='size each image will be resized to for display (width, height)')
    myparser.add_argument('--montage_shape', type=int, nargs=2, default=(4, 4), help='shape of image montage (rows, columns)')
    args=myparser.parse_args()
    input_path=Path(args.inputpath)
    img_list = []
    if input_path.is_dir():
        img_paths=utils.list_images(input_path)
        for img_l in img_paths:
            img = cv2.imread(img_l)
            img_list.append(img)
    else:
        img = cv2.imread(str(args.inputpath))
        num_imgs = args.montage_shape[0] * args.montage_shape[1]
        for i in range(num_imgs):
            img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = build_montages(img_list, args.image_shape,args.montage_shape,single=args.single)
    # iterate through montages and display
    save_dir=Path(args.savedir)
    for i,montage in enumerate(montages):
        cv2.imwrite(str(Path(args.savedir,f"montage{i}.png")), montage)
        cv2.imshow(f"montage{i}", montage)
        cv2.waitKey(0)