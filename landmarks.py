import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt


class FaceLandmarks:
    def __init__(self, img, showImage = False):
        self.showImage = showImage
        self.coords = list()

        # Load the detector
        detector = dlib.get_frontal_face_detector()

        # Load the predictor
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # Convert image into grayscale
        gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

        # Use detector to find landmarks
        faces = detector(gray)
        for face in faces:
            x1 = face.left() # left point
            y1 = face.top() # top point
            x2 = face.right() # right point
            y2 = face.bottom() # bottom point

            # Create landmark object
            landmarks = predictor(image=gray, box=face)

            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                self.coords.append([x,y])

                # Draw a circle
                if self.showImage:
                    cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)


def main():
    img = cv2.imread("images/baby2.jpg")
    landmarks = FaceLandmarks(img, showImage = True)

    cv2.imshow(winname="Face", mat=img)
    cv2.waitKey(delay=0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
