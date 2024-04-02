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
from skimage.io import imread_collection
import torch

# device: str = "mps" if torch.backends.mps.is_available() else "cpu"

model_1 = YOLO("Copy of first_crop.pt")
# model_1.to(device)
# model_2 = YOLO("Copy of Copy of best.pt")
# model_2.to(device)

# characterRecognition = tf.keras.models.load_model("Copy of model_char_recognition_4.h5")
# my_model = YOLO('best_seg.pt')
# my_model.to(device)

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
  # try:
    os.chdir('/home/sobhan/test_proj')
    # detect location of plate
    img1 = 'images/'+img
    result = model_1.predict(source=img1,save_crop = True)
    # change directory to detected and cropd plate
    os.chdir('runs/detect/predict/crops/0')
    # detect characters and numbers
    result = model_2.predict(source=img , save_crop = True)
    result_char = my_model.predict(source=img )
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
    lst_rs=[]
    for i in dic:
      im = cv2.imread(i)
      lst_rs.append(cnnCharRecognition(im))
      # print(cnnCharRecognition(im))
      print(i)
      print("-----------------")
    names = {0: 'a', 1: 'b', 2: 'd', 3: 'eight', 4: 'ein', 5: 'five', 6: 'four',
          7: 'ghaf', 8: 'h', 9: 'jim', 10: 'lam', 11: 'mim', 12: 'nine', 
          13: 'non', 14: 'one', 15: 'sad', 16: 'seven', 17: 'sin', 18: 'six', 19: 'ta', 
          20: 'three', 21: 'two', 22: 'waw', 23: 'wheel', 24: 'y', 25: 'zero'}
    for r in result_char:
      cls_names = r.boxes.cls.tolist() 

    lst = [0,1,2,4,7,8,9,10,11,12,13,14,15,17,19,22,23,24]
    for i in lst :
      if i in cls_names : 
        lst_rs[2] = names[i] 
    # lst_rs[2] =  charac           
    # Deleting an non-empty folder
    dir_path = r"/home/sobhan/test_proj/runs"
    shutil.rmtree(dir_path, ignore_errors=True)
    return(str(lst_rs))

  # except:
  #   dir_path = r"/home/ubuntu/anpr_sj/runs"
  #   shutil.rmtree(dir_path, ignore_errors=True)
  #   return img


# def first_crop(img):
#   try:
#     os.chdir('/home/ubuntu/anpr_sj')
#     # detect location of plate
#     img1 = 'images/'+img
#     result = model_1.predict(source=img1,save_crop = True)
#     # change directory to detected and cropd plate
#     os.chdir('runs/detect/predict/crops/0')
#     # detect characters and numbers
#     result = model_2.predict(source=img , save_crop = True)
#     result_char = my_model.predict(source=img )
#     # change directory to detected and croped characters and numbers
#     os.chdir('runs/detect/predict/crops/number')

#     #access location of detected characters
#     for r in result:
#       crd = r.boxes.xyxy
#     coordinates = crd.tolist()
#     print(coordinates)

#     lst = [img]
#     dic = {img:coordinates[0][0]}
#     for i in range(2,len(coordinates)):
#       cropped_name = img[:-4]+ str(i) +'.jpg'
#       dic[cropped_name] = coordinates[i-1][0]

#     print(dic)
#     #sort croppd characters and numbers by their locations
#     dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}

#     #character recognetion and classification
#     lst_rs=[]
#     for i in dic:
#       im = cv2.imread(i)
#       lst_rs.append(cnnCharRecognition(im))
#       # print(cnnCharRecognition(im))
#       print(i)
#       print("-----------------")
#     names = {0: 'a', 1: 'b', 2: 'd', 3: 'eight', 4: 'ein', 5: 'five', 6: 'four',
#           7: 'ghaf', 8: 'h', 9: 'jim', 10: 'lam', 11: 'mim', 12: 'nine', 
#           13: 'non', 14: 'one', 15: 'sad', 16: 'seven', 17: 'sin', 18: 'six', 19: 'ta', 
#           20: 'three', 21: 'two', 22: 'waw', 23: 'wheel', 24: 'y', 25: 'zero'}
#     for r in result_char:
#       cls_names = r.boxes.cls.tolist() 

#     lst = [0,1,2,4,7,8,9,10,11,12,13,14,15,17,19,22,23,24]
#     for i in lst :
#       if i in cls_names : 
#         lst_rs[2] = names[i] 
#     # lst_rs[2] =  charac           
#     # Deleting an non-empty folder
#     dir_path = r"/home/ubuntu/anpr_sj/runs"
#     shutil.rmtree(dir_path, ignore_errors=True)
#     return(str(lst_rs))
#   except:
#     dir_path = r"/home/ubuntu/anpr_sj/runs"
#     shutil.rmtree(dir_path, ignore_errors=True)
#     return img



# a = first_crop('427700_2.jpg')
# resultt=first_crop('427698_1.jpg')
# result_dic={'numbers':resultt,'img_name':'427698_1.jpg'}
# print(result_dic)