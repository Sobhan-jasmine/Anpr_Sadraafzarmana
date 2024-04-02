# import requests
# url = "http://localhost:8080/files"
# files = {'media': open('10.jpg', 'rb').read()}
# print(requests.post(url, files=files))
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFil
app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent


import base64
image_path = '10.jpg'
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())


print  ( encoded_string   )


import requests

url = "http://37.32.121.233:9209/files"
# headers = {'Content-Type': 'application/json'}
data = {'image': encoded_string.decode('utf-8')}

# send post request
response = requests.post(url, json=data)

# if request is successfull
if response.status_code == 200:
    print('Image uploaded successfully.')
    print(response.text) # or get json response using response.json()
else:
    print('Error uploading image.')    