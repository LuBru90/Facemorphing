try:
    import sys
    import matplotlib.pyplot as plt
    import numpy as np
    import time

    import cv2
    import landmarks
    from affineTransform import getTransformMatrix, transformImage

except ImportError as e:
    print(e)
    print("Try using: pip install -r requirements.txt")
    sys.exit()


class Morph:
    def __init__(self, image1, image2):
        self.image1 = image1
        self.image2 = image2

        if self.image1.shape != self.image2.shape:
            print("Image 1 and Image 2 must have the same dimensions!")
            print("Shape 1:", self.image1.shape)
            print("Shape 2:", self.image2.shape)
            print("Exit")
            sys.exit()

        self.morphedImage = None
        self.triags = None
        self.landmarks = dict()
        self.globalIndex = self._stepCounter()

    def pipeLine(self, alpha):
        t0 = time.time()

        self.alpha = alpha
        self.getCommonLandmarks()
        self.triangulateImages() # Shift alpha for animation
        morphedImage = self.morpheImage()

        t1 = time.time()
        print("---- Spend time: {spendTime:.2f} sec. ---- \n".format(spendTime = t1 - t0))
        return morphedImage

    def _stepCounter(self):
        i = 1
        while 1:
            yield i
            i += 1

    def showPlots(self):
        #self.showNaiveMorphe()
        self.showLandmarks()
        self.showTriangles()
        #self.showMorphedImages()
        self.showFinalImage()
    
    def showNaiveMorphe(self):
        print("Simple alpha blending: 1/5")
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(self.image1)
        ax[1].imshow(np.int32(self.image1 * (1 - self.alpha) + self.image2 * self.alpha))
        ax[2].imshow(self.image2)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.show()

    def showLandmarks(self):
        print("Found landmarks: 2/5")
        x1,y1 = self.landmarks[0][:,0], self.landmarks[0][:,1]
        xm,ym = self.midPoints[:,0], self.midPoints[:,1]
        x2,y2 = self.landmarks[1][:,0], self.landmarks[1][:,1]

        fig, ax = plt.subplots(1,3)
        ax[0].plot(x1, y1, "x", color="red")
        ax[0].imshow(self.image1)

        ax[1].plot(x1, y1, "x", color="red")
        ax[1].plot(xm, ym, "o", color="yellow")
        ax[1].plot(x2, y2, "x", color="blue")
        ax[1].imshow(self.morphedImage)

        ax[2].plot(x2, y2, "x", color="blue")
        ax[2].imshow(self.image2)

        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.show()

    def showTriangles(self):
        print("Delaunay triangulation: 3/5")
        fig, ax = plt.subplots(1,3)
        for index, image in enumerate(["image1", "morphed", "image2"]):
            for key, value in self.triags.items():
                x1, y1 = list(self.triags[key][image][:,0]), list(self.triags[key][image][:,1])
                x1.append(x1[0])
                y1.append(y1[0])
                ax[index].plot(x1, y1, "r")

        ax[0].imshow(self.image1)
        ax[1].imshow(self.morphedImage)
        ax[2].imshow(self.image2)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.show()

    def showMorphedImages(self):
        print("Morphed images: 4/5")
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(self.morphedImage1)
        ax[1].imshow(self.morphedImage)
        ax[2].imshow(self.morphedImage2)
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        plt.show()

    def showFinalImage(self):
        print("Final result: 5/5")
        figure, (ax1, ax2, ax3) = plt.subplots(1,3)
        #m1 = (322, 327)
        #m2 = (298, 345)
        #m3 = (273, 360)
        #padding = 50
        #ax1.imshow(self.image1[m1[1]-padding:m1[1]+padding, m1[0]-padding:m1[0]+padding], cmap="gray")
        #ax2.imshow(self.morphedImage[m2[1]-padding:m2[1]+padding, m2[0]-padding:m2[0]+padding], cmap="gray")
        #ax3.imshow(self.image2[m3[1]-padding:m3[1]+padding, m3[0]-padding:m3[0]+padding], cmap="gray")

        ax1.imshow(self.image1)
        ax2.imshow(self.morphedImage)
        ax3.imshow(self.image2, cmap="gray")
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        plt.show()

    def saveImage(self, image, name, index):
        imageName = "%s_%s.jpg" % (name, index)
        print("Saving image as:", imageName)
        plt.imsave(imageName, np.float32(image/image.max()))

    def morpheImage(self):
        print("%s) Morphing images together:" % next(self.globalIndex))
        self.morphedImage1 = np.zeros(self.image1.shape, dtype = np.int32)
        self.morphedImage2 = np.zeros(self.image2.shape, dtype = np.int32)
        
        for index, triag in self.triags.items():
            for iImage, (image, originalImage) in enumerate(zip(["image1", "image2"], [self.image1, self.image2])):

                # Own implementation:
                # 0.01 sec -> cv2 transformation is faster
                # M = getTransformMatrix(np.float32(triag[image]), np.float32(triag["morphed"]))
                # warpedImage = transformImage(originalImage, M)

                # 0.002 sec 
                M = cv2.getAffineTransform(np.float32(triag[image]), np.float32(triag["morphed"]))
                warpedImage = cv2.warpAffine(originalImage, M, (originalImage.shape[1], originalImage.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)

                triangleMask = np.int32(cv2.fillConvexPoly(np.zeros(originalImage.shape), triag["morphed"], [1,1,1]))       # draws a triangle on an empty canvas
                filledTriangle = triangleMask * warpedImage                                                                 # filles the triangle with the warped image

                if iImage == 0:
                    self.morphedImage1 = np.int32(np.where(self.morphedImage1 == 0, filledTriangle, self.morphedImage1))
                elif iImage == 1:
                    self.morphedImage2 = np.int32(np.where(self.morphedImage2 == 0, filledTriangle, self.morphedImage2))

                """
                if index == 65 and iImage == 1:
                    temp1, mid, temp2  = list(triag["image1"]), list(triag["morphed"]), list(triag["image2"])
                    temp1.append(triag["image1"][0])
                    x1 = list()
                    y1 = list()
                    [x1.append(i[0]) for i in temp1]
                    [y1.append(i[1]) for i in temp1]

                    mid.append(triag["morphed"][0]) 
                    xm = list()
                    ym = list()
                    [xm.append(i[0]) for i in mid]
                    [ym.append(i[1]) for i in mid]

                    temp2.append(triag["image2"][0]) 
                    x2 = list()
                    y2 = list()
                    [x2.append(i[0]) for i in temp2]
                    [y2.append(i[1]) for i in temp2]
                    
                    red = (x1[1], y1[1])
                    yellow = (xm[1], ym[1])
                    blue = (x2[1], y2[1])

                    fig, ax = plt.subplots(1, 3)
                    ax[0].imshow(self.image1)
                    ax[0].plot(x1,y1, "red")
                    ax[0].plot(red[0], red[1], "x", color="red")
                    ax[0].plot(yellow[0], yellow[1], "x", color="yellow")
                    ax[0].plot(blue[0], blue[1], "x", color="blue")

                    ax[1].imshow(np.int32(self.morphedImage1//2) + np.int32(self.morphedImage2//2))
                    ax[1].plot(xm,ym, "yellow")
                    ax[1].plot(red[0], red[1], "x", color="red")
                    ax[1].plot(yellow[0], yellow[1], "x", color="yellow")
                    ax[1].plot(blue[0], blue[1], "x", color="blue")

                    ax[2].imshow(self.image2)
                    ax[2].plot(x2,y2, "blue")
                    ax[2].plot(red[0], red[1], "x", color="red")
                    ax[2].plot(yellow[0], yellow[1], "x", color="yellow")
                    ax[2].plot(blue[0], blue[1], "x", color="blue")
                    plt.show()
                    """

        self.morphedImage = np.int32(self.morphedImage1 * (1 - self.alpha) + self.morphedImage2 * self.alpha)
        return self.morphedImage

    def _fillTriangle(self, image, triangle):
        image = np.int32(np.where(self.morphedImage2 == 0, filledTriangle, self.morphedImage2))

    def getCommonLandmarks(self):
        print("%s) Finding landmarks on given images:" % next(self.globalIndex))
        landmarks1 = landmarks.FaceLandmarks(self.image1).coords
        landmarks2 = landmarks.FaceLandmarks(self.image2).coords
        
        # Remove points from the mouth to avoid artefacts due to the morphing process -> happens when the points are to close together
        #del landmarks1[61:65]
        #del landmarks2[61:65]

        self.landmarks[0] = np.int32(landmarks1)
        self.landmarks[1] = np.int32(landmarks2)
    
    def triangulateImages(self):
        print("%s) Calculating landmark positions of the morphed image: (alpha = %s)" % (next(self.globalIndex),self.alpha))
        midPoints_noFrame = np.int32((1-self.alpha)*self.landmarks[0] + self.alpha * self.landmarks[1])

        ySize = self.image1.shape[0] - 1
        xSize = self.image1.shape[1] - 1
        
        # Add corners and edges of the image to the list
        self.midPoints = np.append(midPoints_noFrame, [
                                                    [0,0],          [xSize//2, 0],      [xSize, 0],
                                                    [0, ySize//2],                      [xSize, ySize//2],
                                                    [0, ySize],     [xSize//2, ySize],  [xSize, ySize]

                                                 ], axis = 0)

        rect = (0, 0, self.image1.shape[1], self.image1.shape[0])

        # Triangulate
        print("%s) Apply delaunay triangulation on these positions:" % next(self.globalIndex))
        subdivision = cv2.Subdiv2D(rect)
        [subdivision.insert((x,y)) for x, y in self.midPoints]

        morphedTriags = np.int32([np.reshape(x, (3,2)) for x in subdivision.getTriangleList()]) # [p1, p2, p3]

        # lut = p1(image1) mid(morphed) p2(image2)
        lut = dict()
        for index, (landmark1, midPoint, landmark2) in enumerate(zip(self.landmarks[0], self.midPoints, self.landmarks[1])):
            lut[index] = np.int32([landmark1, midPoint, landmark2])

        print("%s) Shifting triangles to the landmarks of the input images: (Ensures corresponding triangles)" % next(self.globalIndex))
        # shift triangles to image1 and image2
        self.triags = dict()
        for triagIndex, morphedTriag in enumerate(morphedTriags):
            self.triags[triagIndex] = dict()
            newTriag1 = np.copy(morphedTriag)
            newTriag2 = np.copy(morphedTriag)
            for i, landmark in lut.items():
                p1, mid, p2 = landmark
                for index, triagPoint in enumerate(list(morphedTriag)):
                    comp = triagPoint == mid
                    if comp.all():
                        newTriag1[index] = p1
                        newTriag2[index] = p2

            self.triags[triagIndex]["image1"] = newTriag1
            self.triags[triagIndex]["morphed"] = morphedTriag
            self.triags[triagIndex]["image2"] = newTriag2

        return self.triags

def main():
    image1 = plt.imread("images/lena.jpg")
    image2 = plt.imread("images/monaflip.jpg")

    image1 = plt.imread("images/bond.jpg")
    image2 = plt.imread("images/scarlet.jpg")

    myMorph = Morph(image1, image2)
    newImage = myMorph.pipeLine(alpha = 0.5)
    myMorph.showPlots()

if __name__ == "__main__":
    main()
