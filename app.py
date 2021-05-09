from flask import Flask, jsonify,  request, render_template
import joblib
import numpy as np
from model import pdt_recommendation

app = Flask(__name__)

model_load = pdt_recommendation()

@app.route('/')
def home():
    return render_template('templates/index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        username = request.args['USERNAME']
        top_5_pdt = model_load.predict(str(username)) 
        output = "\n1) {0} \n2) {1} \n3) {2} \n4) {3} \n5) {4}".format(top_5_pdt[0],top_5_pdt[1],top_5_pdt[2],top_5_pdt[3],top_5_pdt[4])
        return render_template('index.html', prediction_text='{top 5 recommendations are : {0}'.format(output))
    else :
        return render_template('index.html')


if __name__ == '__main__':
    app.run()