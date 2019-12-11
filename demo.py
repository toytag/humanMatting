from __future__ import division
import os
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

parser = argparse.ArgumentParser(description='Demo model on images or videos')
parser.add_argument('path', type=str, help='path to an image or a video')
parser.add_argument('--gray', action='store_true',
                    help='segmentation in gray mode. RGB by default.')
args = parser.parse_args()

# check file
if args.path != '0' and not os.path.exists(args.path):
    print('Path:<{}> does not exist'.format(args.path))
    exit()

# pre-defined ratio
presetRatio = np.array([1/2, 9/16, 3/4, 1, 4/3, 16/9, 2])
# these are the sizes where the model performs well
# feel free to change it as long as width and height are multiples of 32
presetSize = [(256, 512), (288, 512), (384, 512),
              (256, 256), (512, 384), (512, 288), (512, 256)]

# load model
if args.gray:
    with open('models/dark.json', 'r') as f:
        json_string = f.read()
    model = model_from_json(json_string)
    model.load_weights('models/dark-28-0.9787.h5')
else:
    with open('models/night.json', 'r') as f:
        json_string = f.read()
    model = model_from_json(json_string)
    model.load_weights('models/night-15-0.9714.h5')

# segmentation
def seg(img, gray=False):
    if gray:
        # convert img to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # aquaire alpha map
        res = np.argmax(
            model.predict(
                img[np.newaxis, :, :, np.newaxis].astype(np.float32)
            ),
            axis=3
        ).reshape(img.shape)
        # apply alpha map
        img[res == 0] = 0
    else:
        # aquaire alpha map
        res = np.argmax(
            model.predict(
                img[np.newaxis, :, :, :].astype(np.float32)
            ),
            axis=3
        ).reshape(img.shape[:-1])
        # apply alpha map
        img[res == 0] = [0, 0, 0]
    return img

# get image or video
img = cv2.imread(args.path)

# image
try:
    ratio = img.shape[1] / img.shape[0]
    ratioIndex = np.argmin(np.abs(ratio - presetRatio))
    img_shape = presetSize[ratioIndex]
    img = cv2.resize(img, img_shape)
    img = seg(img, gray=args.gray)
    cv2.imshow('img', img)
    cv2.waitKey(0)
# video
except:
    cap = cv2.VideoCapture(args.path if args.path != '0' else 0)
    ret, frame = cap.read()
    if ret:
        ratio = frame.shape[1] / frame.shape[0]
        ratioIndex = np.argmin(np.abs(ratio - presetRatio))
        frame_shape = presetSize[ratioIndex]
        frame = cv2.resize(frame, frame_shape)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_shape)
        frame = seg(frame, gray=args.gray)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

cv2.destroyAllWindows()