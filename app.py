from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from keras.models import load_model
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import joblib

# # load the LSTM model
model = load_model(r'models/crude_price.h5')

# load the dataset
train_df = pd.read_csv(r'Dataset/processed/train_df.csv')
scaler = joblib.load('models/scaler.joblib') 

# create Flask app
app = Flask(__name__)

# define route for home page
@app.route('/')
def home():
    return render_template('home.html')

# define route for prediction page
@app.route('/predict', methods=['POST'])

def predict_petrol_price(): 
    try:
        target_date = datetime.strptime(request.form['target_date'], '%Y-%m-%d')        
    except ValueError:
        return render_template('prediction.html', prediction_error='Invalid date format. Please enter date in YYYY-MM-DD format.')
    
    target_date = pd.Timestamp(target_date)  # convert target_date to pd.Timestamp object
    
    # if target_date < train_df["Date"].min():
    if target_date < pd.Timestamp(train_df["Date"].min()):
        return render_template('prediction.html', prediction_error='Target date is earlier than first date in training data')
    
    # target_index = None 
    target_index = len(train_df)  # start by assuming the target date is after the last date in the training data
    for i in range(len(train_df)):
        # if train_df.loc[i, "Date"] >= target_date:
        if pd.Timestamp(train_df.loc[i, "Date"]) >= target_date:
            target_index = i
            break
    # if target_index is None:
    #     raise ValueError("Target date is later than last date in training data") 

    time_steps = 10
    start_index = target_index - time_steps
    if start_index < 0:
        raise ValueError("Not enough data to make prediction")
    sequence = train_df.iloc[start_index:target_index]["Petrol (USD)"].values

    # pad the sequence with zeros if there are less than `time_steps` data points available
    if len(sequence) < time_steps:
        sequence = np.pad(sequence, (time_steps - len(sequence), 0), mode="constant", constant_values=0)

    sequence = sequence.reshape(1, time_steps, 1) 
    predicted_price = model.predict(sequence)[0][0]
    # if scaler == None:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     train_df['Petrol (USD)'] = pd.DataFrame(scaler.fit_transform(train_df.iloc[:, 1:])) 
    #     # scaler.fit(train_df["Petrol (USD)"].values.reshape(-1, 1))  
    #     scaler.fit(train_df.iloc[:, 1:]) 
    predicted_price = scaler.inverse_transform(predicted_price.reshape(1, -1))
    prediction_str = "${:,.2f}".format(predicted_price[0][0])   # format prediction as currency string
    
     # render prediction template with predicted price
    return render_template('prediction.html', predicted_price = prediction_str, target_date = target_date)



if __name__ == '__main__':
    app.run(debug=True)
