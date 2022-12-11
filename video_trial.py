import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def save_frames_to_dir(display=True):
    cap = cv2.VideoCapture('1.mp4')
    
    knt = 0
    while cap.isOpened():
      ret, frame = cap.read()
      
      if not ret:
        print("Unable to read frames!")
        break
    
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      if knt <= 845:
          cv2.imwrite(f"D:/Research/Frames/Play/{knt}.png", gray)
      else:
          cv2.imwrite(f"D:/Research/Frames/Ad/{knt}.png", gray)
    
      knt = knt+1
      if display:
          cv2.imshow('frames', gray)
          if cv2.waitKey(1) == ord('q'):
            break
    
    print(knt)
    cap.release()
    cv2.destroyAllWindows()

# save_frames_to_dir()


path = 'D:/Research/Frames/Train/'

categories = []
for dir in os.listdir(path):
    categories.append(dir)

for category in categories:
    data_dir = f"{path}{category}/"
    for file in os.listdir(data_dir):
        print(file)
        img_dir = f"{data_dir}{file}"
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        # cv2.imshow("Gray", img)
        plt.imshow(img)
        break
    break

SIZE = 128
training_set = []

for i,category in enumerate(categories):
    data_dir = f"{path}{category}/"
    for file in os.listdir(data_dir):
        print(i, file)
        img_dir = f"{data_dir}{file}"
        img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
        img = np.array(cv2.resize(img, (SIZE,SIZE)))
        training_set.append([img , i])
        # X.append(img)
        # y.append(i)
        # cv2.imshow("Gray", img)
        # plt.imshow(img)
        # break
    # break

random.shuffle(training_set)


X = []
y = []

for img_array, label in training_set:
  X.append(img_array)
  y.append(label)
  
X =  np.array(X).reshape(-1,SIZE, SIZE, 1)
y = np.array(y)

import pickle

pickle_out = open('X.pickle' , 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle' , 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()


    

