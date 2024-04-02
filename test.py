# import tensorflow as tf
# from tensorflow.keras import layers, models
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
import tensorflow as tf
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib


# tf.debugging.set_log_device_placement(True)


device: str = "mps" if torch.backends.mps.is_available() else "cpu"

model_11 = YOLO("Copy of first_crop.pt")
# model_1.to('cuda')
model_11.to(device)
model_2 = YOLO("new_weight_50_best.pt")
model_2.to(device)

# characterRecognition = tf.keras.models.load_model("Copy of model_char_recognition_4.h5")
my_model = YOLO('best_seg.pt')
my_model.to(device)

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion

# Create model_1 and send it to the target device
torch.manual_seed(42)
model_1 = TinyVGG(
    input_shape=3,
    hidden_units=10,
    output_shape=10).to(device)

model_1.load_state_dict(torch.load(r'number_recognation_fuladyar.pth'))


# def cnnCharRecognition(img):
#     dictionary = {0:'0', 1:'1', 2 :'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 10:'A',
#     11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I', 19:'J', 20:'K',
#     21:'L', 22:'M', 23:'N', 24:'P', 25:'Q', 26:'R', 27:'S', 28:'T', 29:'U',
#     30:'V', 31:'W', 32:'X', 33:'Y', 34:'Z'}

#     blackAndWhiteChar=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blackAndWhiteChar = cv2.resize(blackAndWhiteChar,(75,100))
#     image = blackAndWhiteChar.reshape((1,100,75,1))
#     image = image / 255.0
#     new_predictions = characterRecognition.predict(image)
#     char = np.argmax(new_predictions)
#     return char

def number_recog(custom_image_path):
  # Load in custom image and convert the tensor values to float32
  custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)

  # Divide the image pixel values by 255 to get them between [0, 1]
  custom_image = custom_image / 255.
  # Create transform pipleine to resize image
  custom_image_transform = transforms.Compose([
      transforms.Resize((64, 64)),
  ])

  # Transform target image
  custom_image_transformed = custom_image_transform(custom_image)
  model_1.eval()
  with torch.inference_mode():
      # Add an extra dimension to image
      custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

      # Print out different shapes
      print(f"Custom image transformed shape: {custom_image_transformed.shape}")
      print(f"Unsqueezed custom image shape: {custom_image_transformed_with_batch_size.shape}")

      # Make a prediction on image with an extra dimension
      custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
  # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
  custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
  # Convert prediction probabilities -> prediction labels
  custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
  class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  # Find the predicted label
  custom_image_pred_class = class_names[custom_image_pred_label.cpu()] # put pred label to CPU, otherwise will error
  return custom_image_pred_class


def first_crop(img):
      # try:
    os.chdir('/home/sobhan/test_proj')
    # detect location of plate
    img1 = 'images/'+img
    result = model_11.predict(source=img1,save_crop = True)
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
      custom_image_path = i
      lst_rs.append(number_recog(custom_image_path))
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
#//////////////////////////////////////////////////////////////////////////////////////////////// 
    # #access location of detected characters
    # for r in result:
    #   crd = r.boxes.xyxy
    # coordinates = crd.tolist()
    # print(coordinates)

    # lst = [img]
    # dic = {img:coordinates[0][0]}
    # for i in range(2,9):
    #   cropped_name = img[:-4]+ str(i) +'.jpg'
    #   dic[cropped_name] = coordinates[i-1][0]

    # print(dic)
    # #sort croppd characters and numbers by their locations
    # dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}

    # #character recognetion and classification
    # lst_rs=[]
    # for i in dic:
    #   im = cv2.imread(i)
    #   lst_rs.append(cnnCharRecognition(im))
    #   # print(cnnCharRecognition(im))
    #   print(i)
    #   print("-----------------")
    # names = {0: 'a', 1: 'b', 2: 'd', 3: 'eight', 4: 'ein', 5: 'five', 6: 'four',
    #       7: 'ghaf', 8: 'h', 9: 'jim', 10: 'lam', 11: 'mim', 12: 'nine', 
    #       13: 'non', 14: 'one', 15: 'sad', 16: 'seven', 17: 'sin', 18: 'six', 19: 'ta', 
    #       20: 'three', 21: 'two', 22: 'waw', 23: 'wheel', 24: 'y', 25: 'zero'}
    # for r in result_char:
    #   cls_names = r.boxes.cls.tolist() 

    # lst = [0,1,2,4,7,8,9,10,11,12,13,14,15,17,19,22,23,24]
    # for i in lst :
    #   if i in cls_names : 
    #     lst_rs[2] = names[i] 
    # # lst_rs[2] =  charac           
    # # Deleting an non-empty folder
    # dir_path = r"/home/sobhan/test_proj/runs"
    # shutil.rmtree(dir_path, ignore_errors=True)
    # return(str(lst_rs))
#////////////////////////////////////////////////
    # return result




# a = first_crop('411973_2.jpg')
# resultt=first_crop('411973_2.jpg')
# result_dic={'numbers':resultt,'img_name':'411973_2.jpg'}
# print(result_dic)
# im=first_crop()
# print(im)
# img = cv2.imread(r'/home/sobhan/test_proj/411973_2.jpg')
# # a = first_crop(img)
# resultt=first_crop(img)

