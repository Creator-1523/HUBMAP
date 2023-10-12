from flask import Flask, render_template,request,flash,redirect
import joblib
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import keras
import os

ALLOWED_EXTENSIONS = set(['tif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app=Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
PREDICTION_FOLDER = 'static/predict/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREDICTION_FOLDER'] = PREDICTION_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# model=joblib.load('model.pkl')
model = keras.models.load_model('model.keras')
colourmap=[
    [0,0,0],
    [255,0,0],
    [0,255,0]
]

CLASSES=["One","Two","Three"]
def grayscale_to_rgb(mask, classes, colormap):
    h=512
    w=512
    mask = mask.astype(np.int32)
    output = []

    for i, pixel in enumerate(mask.flatten()):
        output.append(colormap[pixel])
    print(np.array(output).shape)
    output = np.reshape(output, (h, w, 3))
    return output

def save_results(image, mask, pred, save_image_path):
    h, w, _ = image.shape
    line = np.ones((h, 10, 3)) * 255
    
    pred = np.expand_dims(pred, axis=-1)
    pred = grayscale_to_rgb(pred, CLASSES, colourmap)

#     cat_images = np.concatenate([image, line, mask, line, pred], axis=1)
    cv2.imwrite(save_image_path, pred)

def read_image_mask(x, y):
    """ Reading """
    print(x,y)
    x = cv2.imread(x, cv2.IMREAD_COLOR)
    y = cv2.imread(y, cv2.IMREAD_COLOR)
    assert x.shape == y.shape

    """ Resizing """
    x = cv2.resize(x, (512, 512))
    y = cv2.resize(y, (512, 512))

    """ Image processing """
    x = x / 255.0
    x = x.astype(np.float32)
    """ Mask processing """
    output = []
    for color in colourmap:
        cmap = np.all(np.equal(y, color), axis=-1)
        output.append(cmap)
    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)
    return x, output
@app.route('/',methods=['GET','POST'])
def home():
    if request.method=="POST":
        print("Hi")
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            image = cv2.imread(f"static/uploads/{filename}", cv2.IMREAD_COLOR)

            # image = cv2.resize(image, (IMG_W, IMG_H))
            image_x = image
            image = image/255.0
            image = np.expand_dims(image, axis=0)

            """ Reading the mask """
            # mask = cv2.imread(f"train/truth/{files[n]}.png", cv2.IMREAD_COLOR)

            """ Prediction """
            pred = model.predict(image, verbose=0)[0]
            # pred = np.argmax(pred, axis=-1)
            pred = pred.astype(np.float32)
            pred=pred*255
            cv2.imwrite("static/prediction/test3.png",pred)
        return render_template('index.html',result=1,file=filename.replace('.tif','.png'))
    return render_template("index.html",result=0)

if __name__=="__main__":
    app.run(debug=True)