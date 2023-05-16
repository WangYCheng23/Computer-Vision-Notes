# import argparse
# import cv2
# parser = argparse.ArgumentParser()
# parser.add_argument("path_image", help="path to input image to be displayed")

# rgs = parser.parse_args()
# image = cv2.imread(args.path_image)
# args = vars(parser.parse_args())
# image2 = cv2.imread(args["path_image"])

# cv2.imshow("loaded image", image)
# cv2.imshow("loaded image2", image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("index_camera", default=0, help="index of the camera to read from", type=int)
args = parser.parse_args()

capture = cv2.VideoCapture(args.index_camera)
if capture.isOpened()is False:
    print("Error opening the camera")
    
while capture.isOpened():
    ret, frame = capture.read()

    if ret is True:
        cv2.imshow('Input frame from the camera', frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale input camera', gray_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
capture.release()
cv2.destroyAllWindows()


