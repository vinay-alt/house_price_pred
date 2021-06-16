from flask import Flask, request, jsonify, render_template, url_for, redirect, flash
import json
import pickle
import numpy as np

__columns = None
__locations = None
__model = None
msg = None

app = Flask(__name__)
app.secret_key = 'error'

@app.route("/")
def home():
    global __columns
    global __locations
    global __model

    with open('artifacts/columns.json', 'r') as f:
        __columns = json.load(f)['data_columns']
        __locations = __columns[3:]
    print(msg)
    if (msg == "Something went wrong, Please Check if u have filled all inputs"):
        return render_template("index.html", locations=__locations, msg=msg)
    else:
        return render_template("index.html", locations=__locations)

@app.route("/predict", methods=['GET','POST'])
def predict():
    try:
        loc_index = __columns.index(request.form['loc'])
        x = np.zeros(len(__columns))
        x[0] = request.form['area']
        x[1] = request.form['bhk']
        x[2] = request.form['bath']
        if (loc_index >= 0):
            x[loc_index] = 1

        with open('artifacts/banglore_home_prices_model.pickle', 'rb') as f:
            __model = pickle.load(f)

        predicted_score = round(__model.predict([x])[0], 2)

        return render_template("predict.html", predicted_score=predicted_score)
    except:
        flash("Something went wrong, Please Check if u have filled all inputs")
        return redirect("/")

if __name__=="__main__":
    app.run(debug=True)