import flask
from flask import Flask, render_template, request, Markup
from flask import jsonify
import numpy as np
from keras.models import load_model
import pickle

classifier = load_model('Trained_model.h5')
classifier._make_predict_function()

crop_recommendation_model_path = 'Crop_Recommendation.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)


@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")


@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")


@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        json_data = flask.request.json
        N = json_data.get('nitrogen')
        P = json_data.get('phosphorous')
        K = json_data.get('potassium')
        ph = json_data.get('ph')
        rainfall = json_data.get('rainfall')
        temperature = json_data.get('temperature')
        humidity = json_data.get('humidity')
        #N = request.form.get('nitrogen')
        #P = request.form.get('phosphorous')
        #K = request.form.get('potassium')
        #ph = request.form.get('ph')
        #rainfall = request.form.get('rainfall')
        #temperature = request.form.get('temperature')
        #humidity = request.form.get('humidity')
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        response = {
                    "crop": final_prediction
                }
        return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
