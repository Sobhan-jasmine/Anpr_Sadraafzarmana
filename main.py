from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
# from Anpr import read_imgs
# from Anpr import first_crop
from test import first_crop
import cv2
# from cv2  import dnn_superres
import torch
from skimage.io import imread_collection
from ultralytics import YOLO
from cv2 import dnn_superres
import uvicorn
from typing import List
from pathlib import Path
from fastapi.responses import FileResponse
from ultralytics import YOLO
import tensorflow as tf

import os


device: str = "mps" if torch.backends.mps.is_available() else "cpu"

model_1 = YOLO("Copy of first_crop.pt")
model_1.to(device)
# model_2 = YOLO("Copy of Copy of best.pt")
# model_2.to(device)


# characterRecognition = tf.keras.models.load_model("Copy of model_char_recognition_4.h5")
# my_model = YOLO('best_seg.pt')
# my_model.to(device)
# yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# my_model = YOLO('weights_segment/best.pt')
IMAGEDIR = "images/"
 
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(Path(BASE_DIR, 'templates')))
# templates = Jinja2Templates(directory="templates")
app.mount("/images", StaticFiles(directory="images"), name="images")
 
@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
 

@app.post("/images")
def images(img1: bytes = File(...), img2: bytes = File(...)):
    # **do something**
    return {"message": "OK"}



@app.post("/upload-files")
async def create_upload_files(request: Request, files: List[UploadFile] = File(...)):
    for file in files:
        contents = await file.read()
        #save the file
        os.chdir('/home/sobhan/test_proj')
        name = file.filename
        with open(f"{IMAGEDIR}{file.filename}", "wb") as f:
            f.write(contents)
 
    # show = [file.filename for file in files]
    # resultt  = read_imgs(name)
    
   
    return show_result(Request,name)
    #return {"Result": "OK", "filenames": [file.filename for file in files]}
    # return resultt



@app.get("/show-files",response_class=HTMLResponse)
def show_result(request: Request,name):
    resultt=first_crop(name)
    result_dic={'numbers':resultt,'img_name':name}
    # return templates.TemplateResponse("show.html" ,{"request":Request,"res":result_dic})
    # return templates.TemplateResponse("show.html" ,{"request":Request,"resultt":resultt,"name":name})
    return resultt
    
# if __name__ == '__main__':
#     uvicorn.run(app, port=8000,host='127.0.0.1')