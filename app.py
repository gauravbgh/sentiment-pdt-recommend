from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from model import pdt_recommendation

app = Flask(__name__)

model_load = pdt_recommendation()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        username = request.form.get("USERNAME")
        top_5_pdt = model_load.predict(str(username)) 

        return render_template('index.html', prediction_text= top_5_pdt )
    else :
        return render_template('index.html')


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)