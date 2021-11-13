#import libraries
import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model_rf_t.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
     #For rendering results on HTML GUI
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    data_web = request.form
    data_web = data_web.to_dict()
    data_df = pd.DataFrame(data = data_web, index=[0])
    prediction = model.predict(data_df)
   # proba = model.predict_proba(data_df)
    #output = round(proba[0], 2) 
    return render_template('index.html', prediction_text=' This customer will answer :{}'.format(prediction))

if __name__ == '__main__':
    app.run(debug=True)


