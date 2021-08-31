import numpy as np
import matplotlib.pyplot as plt
import cv2
import time


def getTransformMatrix(origin, destination):
    x = np.zeros(origin.shape[0] + 1) # insert [0]-element for better indexing -> x[1] = first element
    x[1:] = origin[:,0]
    y = np.copy(x)
    y[1:] = origin[:,1]

    x_ = np.copy(x)
    x_[1:] = destination[:,0]
    y_ = np.copy(x)
    y_[1:] = destination[:,1]

    a11 = (y[1] * (x_[2] - x_[3])               + y[2] * (x_[3] - x_[1])              + y[3] * (x_[1] - x_[2]))
    a12 = (x[1] * (x_[3] - x_[2])               + x[2] * (x_[1] - x_[3])              + x[3] * (x_[2] - x_[1]))
    a21 = (y[1] * (y_[2] - y_[3])               + y[2] * (y_[3] - y_[1])              + y[3] * (y_[1] - y_[2]))
    a22 = (x[1] * (y_[3] - y_[2])               + x[2] * (y_[1] - y_[3])              + x[3] * (y_[2] - y_[1]))
    a13 = (x[1] * (y[3]*x_[2] - y[2]*x_[3])     + x[2] * (y[1]*x_[3] - y[3]*x_[1])    + x[3] * (y[2]*x_[1] - y[1]*x_[2]))
    a23 = (x[1] * (y[3]*y_[2] - y[2]*y_[3])     + x[2] * (y[1]*y_[3] - y[3]*y_[1])    + x[3] * (y[2]*y_[1] - y[1]*y_[2]))

    d = x[1]*(y[3] - y[2]) + x[2]*(y[1] - y[3]) + x[3]*(y[2] - y[1])
    return  1/d * np.array([[a11, a12, a13], [a21, a22, a23], [0, 0, 1]])

def transformImage(image, M):
    warpedImage = np.zeros(image.shape, dtype=np.int32)
    for y, row in enumerate(image):
        for x, value in enumerate(row):
            newX, newY, _ = np.dot(M, np.array([x,y,1]))
            cond1 = newY < warpedImage.shape[0] and newX < warpedImage.shape[1]
            cond2 = newY > 0 and newX > 0
            if cond1 and cond2:
                warpedImage[int(newY)][int(newX)] = value
    return warpedImage

def interpolateMissingPixels(image):
    #interpImage = np.zeros(image.shape, dtype=np.int32)
    interpImage = np.array(image)
    for y in range(1, len(image) - 1):
        row = interpImage[y]
        for x in range(1, len(row) - 1):
            if row[x].all() == 0: # empty pixel
                windowPixels = interpImage[y-1:y+2, x-1:x+2]  # [rgb], [rgb], [rgb]

               # if windowPixels.sum() == 0:
               #     continue

                newPixel = np.array([0,0,0])
                for channel in range(3): # interpolate rgb
                    channelValues = windowPixels[:, :, channel]
                    temp = channelValues != 0
                    meancount = temp.sum()
                    newPixel[channel] = channelValues.sum() / meancount if meancount != 0 else 0
                interpImage[y][x] = newPixel
    return interpImage

def main():
    origin      = np.array([[50, 50],  [50, 100], [100, 50]])
    destination = np.array([[50, 100],  [100, 250], [150, 50]])

    m = getTransformMatrix(origin, destination)

    image = plt.imread("scarlet.jpg")[100:400, 100:400]

    warpedImage = transformImage(image, m)
    interpImage = interpolateMissingPixels(warpedImage)

    fig, ax = plt.subplots(1,3)
    ax[0].imshow(image)
    ax[1].imshow(warpedImage)
    ax[2].imshow(interpImage)
    plt.show()

if __name__ == "__main__":
    main()
