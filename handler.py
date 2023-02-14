from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann
import pandas as pd
import pickle
import os

#loading model
model = pickle.load(open('../artifacts/model.pkl', 'rb'))

#initialize api
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    #checking if there is data
    if test_json:
        if isinstance(test_json, dict):
            #unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            #multiple examples
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        #instanciando classe Rossmann
        pipeline = Rossmann()
        
        #data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        #feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        #data preparation
        df3 = pipeline.data_preparation(df2)

        #feature selection
        df = pipeline.feature_selection(df3)
        
        #prediction
        response = pipeline.get_prediction(model, test_raw, df)
        
        return response
    
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)