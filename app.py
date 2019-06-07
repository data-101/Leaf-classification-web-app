from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
app = Flask(__name__)

page=["hibiscus.html","neem.html","tulsi"]
model=load_model("model/leafClassifier_v3.h5")
graph = tf.get_default_graph()

@app.route('/')
def entry_point():
   return render_template('home.html')


@app.route('/uploader', methods = ['POST'])
def upload_file():
   if request.method == 'POST':
      print("debug 0")
      
      im = image.load_img(request.files['file'], target_size=(215, 215))
      img = image.img_to_array(im)
      img = np.expand_dims(img, axis = 0)
      #print(test)
      with graph.as_default():
         result = model.predict(img)
      render_template("plants/"+page[result.argmax()])
      
      return  render_template("plants/"+page[result.argmax()])
		
if __name__ == '__main__':
   app.run(debug = True)#debug = True)