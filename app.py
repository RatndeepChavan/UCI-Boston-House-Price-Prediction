from flask import Flask, render_template, request
import pickle
import numpy as np
import random, threading, webbrowser

app = Flask(__name__)

model = pickle.load(open('Prediction_model', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    fet = [float(x) for x in request.form.values()]
    final = np.array(fet)
    final = final[np.newaxis,:]
    pred = model.predict(final)
    print(pred)
    
    return render_template('home.html', prediction_text=f"House price is : {pred}")

if __name__ == "__main__":
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:{0}".format(port)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)