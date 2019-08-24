import os
import numpy as np
import keras
from keras.applications import imagenet_utils

from flask import Flask, request, render_template, send_from_directory
from keras.preprocessing import image

__author__ = 'Morteza-Waskasi'

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/analysis')
def analysispage():
    return render_template('analysis.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')


@app.route("/")
def index():
    return render_template("upload.html")



@app.route("/results", methods=["POST","GET"])
#def resultspage():

#@app.route("/upload", methods=["POST","GET"])
def upload():
    print ("input",'images/' )
    target = os.path.join(APP_ROOT, 'images/')
    print('target=',target)
    print('this is the type of input:',type(request.files.getlist("file")))
    if isinstance(request.files.getlist("file"), list):
        print("this is a list")
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    print(type(request.files.getlist("file")))
    for upload in request.files.getlist("file"):
        print('this is upload',upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        print('filename type',type(filename))
        destination = "/".join([target, filename])
        print('destination type',type(destination))
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        img=image.load_img(destination, target_size=(128, 128))
        print ('type img',type(img))
        im2=image.img_to_array(img)
        print('shape of im2',np.shape(im2))
        im3 = np.expand_dims(im2, axis=0)
        print('shape of im3',np.shape(im3))
        im4 = imagenet_utils.preprocess_input(im3)
        print('shape of im4',np.shape(im4))
        model = keras.models.load_model('bin_30k_weightslr0001_sigmoid.h5') 
        model.summary()
        classes=model.predict(im4, batch_size=32)
        print('Normal',classes[0][0])
        print('Abnormal',classes[0][1])
        if classes[0][0] >classes[0][1]:
            return render_template("results.html", image_name=filename)
        else:
            return render_template("resultsP.html", image_name=filename)
    # return send_from_directory("images", filename, as_attachment=True)
   # return render_template("complete.html", image_name=filename)

@app.route('/analysis/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

#@app.route('/upload/<filename>')
#def send_image(filename):
#    return send_from_directory("images", filename)

@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
