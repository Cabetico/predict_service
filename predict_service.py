from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import pickle



model_file = 'model_xgb.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

### creating app/microservice
app = Flask('predict')

@app.route('/predict', methods=['POST'])
def predict():


    ##GET POST REQUEST USING REQUEST MODULE
    customer = request.get_json()

    X = dv.transform([customer])
    dexample = xgb.DMatrix(X, feature_names=dv.get_feature_names())
    y_pred = model.predict(dexample)
    
    result = {
        "default_probability": float(y_pred),
        #"label": np.where( np.array(y_pred) < 0.5, "APPROVED", "REJECTED")
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
