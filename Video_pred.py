# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 19:54:19 2022

@author: Chirag
"""

import cv2
import tensorflow as tf
import os

def prepare(img_array):
  IMG_SIZE = 128
  # img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
  new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("Video-CNN.model")

CATEGORIES = ["Ad", "Play"]

cap = cv2.VideoCapture('Football_Copy.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
output = cv2.VideoWriter('D:/Research/001.avi', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
  ret, frame = cap.read()
  
  if not ret:
    print("Unable to read frames!")
    break

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  prediction = model.predict([prepare(gray)])
  text = CATEGORIES[int(prediction[0][0])]

  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(gray, 
                f"Prediction: {text}", 
                (50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
  cv2.imshow('Output', gray)
  output.write(gray)
  if cv2.waitKey(1) == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()



# test_dir = "D:/Research/Football Copy.mp4"
# # preds = []
# for file in os.listdir(test_dir):
#     img_dir = os.path.join(test_dir, file)
    
    # preds.append(CATEGORIES[int(prediction[0][0])])
    

















# from collections import Counter

# img = []
# test_dir = "D:/Research/Frames/Test"
# for file in os.listdir(test_dir):
#     img_dir = os.path.join(test_dir, file)
#     img.append(cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE))

# print(img[1].shape)
# height,width = img[1].shape

# video = cv2.VideoWriter('Video.avi',-1,1,(width,height))

# for i in img:
#     video.write(i)

# cv2.destroyAllWindows()
# video.release()
# print(Counter(preds))




# prediction = model.predict([prepare("D:/Research/Frames/Test/Test_Play/317.png")])
# print(CATEGORIES[int(prediction[0][0])])