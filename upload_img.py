from fastapi import FastAPI, File, UploadFile

import uvicorn
app = FastAPI()


@app.post("/files")
async def UploadImage(file: bytes = File(...)):
    with open('image.jpg','wb') as image:
        image.write(file)
        image.close()
    return 'got it'

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
    