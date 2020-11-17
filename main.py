import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from scipy import ndimage
from scipy import signal
from math import sqrt

# parameters for the subimages
WIDTH = 64
HEIGHT = 128
Dy = 32
Dx = 32
SIZE = (WIDTH, HEIGHT)
STRIDE = (Dy, Dx)
# THRESHOLD = 0.35

def getImagesPaths(file_path: str):
    with open(file_path, 'r') as fh:
        names = []
        line_index = 0
        for line in fh:
            if line_index >= int(os.getenv('NUMIMAGES')):
                break
            names.append(line.strip())
            line_index += 1
        return names

def readAndGrey(path_to_image: str):
    # read the image, convert it to greyscale
    colored_image = cv2.imread(path_to_image)
    grey_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)
    return grey_image

def turnIntoYCbCr(image: np.ndarray):
    # map using hsv
    image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)

    # convert to rgb
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    # convert to YCrCb
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    # we return the image where each element is a float instead of an integer
    # to enable normalization
    return image.astype(np.float32)

def createSubImage(image: np.ndarray, mask: np.ndarray):
    # find the location of the plate according to the mask
    best_sub_image = np.array(SIZE)
    best_sub_image_ratio = 0
    for y in range(0, image.shape[1], Dy):
        for x in range(0, image.shape[0], Dx):
            try:
                # get a portion of the mask to compare it
                sub_image = mask[ x: x+WIDTH, y: y+HEIGHT ]
                
                # we get the number of plate pixels inside the sub_image
                plate_pixels =  np.sum(sub_image == 255)

                # we get the number of pixels inside the sub_image (usually 3000, given width 100 and height 30)
                total_pixels = sub_image.shape[0] * sub_image.shape[1]

                ratio = plate_pixels/total_pixels
                # if the number of white pixels is great enough...
                # we find it in the original image and return it
                if ratio > best_sub_image_ratio and sub_image.shape == SIZE:
                    best_sub_image = image[ x: x+WIDTH, y: y+HEIGHT ]
                    best_sub_image_ratio = ratio
                
                
            except:
                #in case the window exceeds the image's dimensions
                pass

    # # to print the first figure referenced in the report
    # plt.figure()
    # plt.imshow(best_sub_image)
    # plt.savefig('./output/patches/1.png', bbox_inches='tight')
    # plt.close()

    best_sub_image = turnIntoYCbCr(best_sub_image)

    # # to print the second figure referenced in the report
    # plt.figure()
    # plt.imshow(best_sub_image)
    # plt.savefig('./output/patches/2.png', bbox_inches='tight')
    # plt.close()

    return best_sub_image

def batchNormalization(image: np.ndarray):
    # iterate through the color channels (j = 0,1,2)

    for j in range(image.shape[2]):
        # take all the pixels of the j channel as a 2D list and apply batch norm
        s = image[:, :, j].shape[0] * image[:, :, j].shape[1]
        mean = np.sum(image[:, :, j]) / s
        deviation = sqrt(( np.sum( (image[:, :, j] - mean)**2 ) ) / (s - 1))

        # now we normalize
        image[:, :, j] = (image[:, :, j] - mean)/deviation

    # # to print the third figure referenced in the report
    # plt.figure()
    # plt.imshow(image)
    # plt.savefig('./output/patches/3.png', bbox_inches='tight')
    # plt.close()

    return image


# kernel bank creation
class KernelBank:
    def __init__(self, shape: int, size:int):
        self.kernels = []
        self.results = []
        for i in range(size):

            # # .seed makes every kernel be exactly the same (to provide some predictability)
            # seed = 0 was used to generate figure 4 of the report
            # np.random.seed(0)
            kernel = np.random.rand( shape, shape, shape )
            self.kernels.append( kernel )

    def convolve(self, image) -> list:
        for i in range(len(self.kernels)):
            # print("KERNEL: " + str(i))
            # print(self.kernels[i])

            # convolve is faster than convolve2d
            result = ndimage.convolve(image, self.kernels[i], mode='constant', cval=0.0)

            # # norm equal to one wields less variations between kernels (only one channel is considered)
            # result0 = signal.convolve2d(image[:,:,0], self.kernels[i][:,:,0], mode='same')
            # result1 = signal.convolve2d(image[:,:,1], self.kernels[i][:,:,0], mode='same')
            # result2 = signal.convolve2d(image[:,:,2], self.kernels[i][:,:,0], mode='same')
            # result = np.zeros( (WIDTH, HEIGHT, 3) )
            # result[:,:,0] = result0
            # result[:,:,1] = result1
            # result[:,:,2] = result2
            
            #now we apply ReLU and append it to the results
            self.results.append( np.maximum(result, 0) )
        return self.results


if __name__ == "__main__":

    # get the first x images (according to the NUMIMAGES enviroment variable)
    imagesPaths = getImagesPaths('./train1.txt')
    masksPaths = [ i.replace('orig', 'mask') for i in imagesPaths ]

    # clean the output
    if os.path.exists('./output/patches'):
        shutil.rmtree('./output/patches', ignore_errors=True)
    if not os.path.exists('./output/patches'):
        os.makedirs('./output/patches')


    for img, msk in zip(imagesPaths, masksPaths):
        # each image is actually a ndarray
        name = img.split('/')[2].split('.')[0]
        image = readAndGrey(img)
        mask = readAndGrey(msk)

        sub_image = createSubImage(image, mask)

        # now we can batch normalize the image (which is in the YCbCr color space)
        sub_image = batchNormalization(sub_image)

        # we create our kernels just 1 filter (3x3)
        kb = KernelBank(3, int(os.getenv('NUMFILTERS')))

        # we convolve our image with the KernelBank and also apply ReLU
        results = kb.convolve(sub_image)

        # save each result of the convolution
        for i in range(len(results)):
            plt.figure()
            plt.imshow(results[i])
            plt.savefig('./output/patches/'+ name + '_' + str(i) +'.png', bbox_inches='tight')
            plt.close()

        # now we convolve with a sobel, for comparison's sake
        sobelY = np.array([
            [[1,2,1],[0,0,0],[-1,-2,-1]],
            [[1,2,1],[0,0,0],[-1,-2,-1]],
            [[1,2,1],[0,0,0],[-1,-2,-1]]
        ])

        # result = ndimage.convolve(sub_image, sobelY, mode='constant', cval=0.0)
        result = sub_image

        # #2d convolutions of each channel
        # sobelY = np.array([
        #     [1,2,1],
        #     [0,0,0],
        #     [-1,-2,-1]
        # ])

        # result0 = signal.convolve2d(sub_image[:,:,0], sobelY, mode='same')
        # result1 = signal.convolve2d(sub_image[:,:,1], sobelY, mode='same')
        # result2 = signal.convolve2d(sub_image[:,:,2], sobelY, mode='same')
        # result = np.zeros( (WIDTH, HEIGHT, 3) )
        # result[:,:,0] = result0
        # result[:,:,1] = result1
        # result[:,:,2] = result2
 
        # this is figure 5 in the report
        plt.figure()
        plt.imshow(result)
        plt.savefig('./output/patches/'+ name + '_sobel.png', bbox_inches='tight')
        plt.close()