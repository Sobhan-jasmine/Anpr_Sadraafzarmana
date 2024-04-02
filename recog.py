from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import cv2
import os
from ultralytics import YOLO
import shutil


model_1 = YOLO("Copy of first_crop.pt")
model_2 = YOLO("Copy of Copy of best.pt")

characterRecognition = tf.keras.models.load_model("Copy of model_char_recognition_4.h5")

def cnnCharRecognition(img):
    dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
    11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
    21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
    30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

    blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
    image = blackAndWhiteChar.reshape((1,100,75,1))
    image = image / 255.0
    new_predictions = characterRecognition.predict(image)
    char = np.argmax(new_predictions)
    return char




def first_crop(img):
  # detect location of plate
  result = model_1.predict(source=img, save_crop = True)
  # change directory to detected and cropd plate
  os.chdir('runs/detect/predict/crops/0')
  # detect characters and numbers
  result = model_2.predict(source=img , save_crop = True)
  # change directory to detected and croped characters and numbers
  os.chdir('runs/detect/predict/crops/number')

  #access location of detected characters
  for r in result:
    crd = r.boxes.xyxy
  coordinates = crd.tolist()
  print(coordinates)

  lst = [img]
  dic = {img:coordinates[0][0]}
  for i in range(2,9):
    cropped_name = img[:-4]+ str(i) +'.jpg'
    dic[cropped_name] = coordinates[i-1][0]

  print(dic)
  #sort croppd characters and numbers by their locations
  dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}

  #character recognetion and classification
  for i in dic:
    im = cv2.imread(i)
    print(cnnCharRecognition(im))
    print(i)
    print("-----------------")
  # Deleting an non-empty folder
  dir_path = r"/home/sobhan/test_proj/runs"
  shutil.rmtree(dir_path, ignore_errors=True)



first_crop('test10.jpg')