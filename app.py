from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('pi_model.h5')

class_names = [
    'Moon jellyfish (Aurelia aurita)',
    'Barrel jellyfish (Rhizostoma pulmo)',
    'Blue jellyfish (Cyanea lamarckii)',
    'Compass jellyfish (Chrysaora hysoscella)',
    'Lion’s mane jellyfish (Cyanea capillata)',
    'Mauve stinger (Pelagia noctiluca)'
]

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

uploaded_image = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global uploaded_image
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        img = Image.open(file)
        uploaded_image = img.copy()

        img = prepare_image(img, target_size=(224, 224))
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_names[predicted_class]

        return render_template('index.html', prediction=predicted_label)
    return render_template('index.html')

@app.route('/display_image')
def display_image():
    global uploaded_image
    if uploaded_image is None:
        return '', 204
    img_io = io.BytesIO()
    uploaded_image.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
