import numpy as np
import pandas as pd 

def treat_missing_numeric(df,columns,how = 'mean', value = None):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mean', 'mode', 'median','ffill', numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'ffill':
        for i in columns:
            print("Filling missing values with forward fill for columns - {0}".format(i))
            df[i] = df[i].fillna(method ='ffill')
    
    elif how == 'digit':
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how, i))
            df[i] = df[i].fillna(str(value)) 
      
    else:
        print("Missing value fill cannot be completed")
    return df


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step):
        a = dataset[i:(i+time_step), 1]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 1])
    return np.array(dataX), np.array(dataY)


def save_model(model, filename):
    """
    Save a trained scikit-learn model to disk using joblib.
    """
    try: 
        model.save(filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model to {filename}: {e}")
        

def predict_petrol_price(train_df, target_date, scaler, model:str): 
    # target_date = pd.to_datetime(target_date)
    try:
        target_date = pd.to_datetime(target_date)
    except ValueError:
        return "Invalid date format. Please enter date in YYYY-MM-DD format."
    
    # if target_date < train_df["Date"].min():
    #     raise ValueError("Target date is earlier than first date in training data")
    
    if target_date < train_df["Date"].min():
        return "Target date is earlier than first date in training data"
    
    target_index = len(train_df)  # start by assuming the target date is after the last date in the training data
    
    for i in range(len(train_df)):
        if train_df.loc[i, "Date"] >= target_date:
            target_index = i
            break

    time_steps = 10
    start_index = target_index - time_steps
    if start_index < 0:
        raise ValueError("Not enough data to make prediction")
    sequence = train_df.iloc[start_index:target_index]["Petrol (USD)"].values

    # pad the sequence with zeros if there are less than `time_steps` data points available
    if len(sequence) < time_steps:
        sequence = np.pad(sequence, (time_steps - len(sequence), 0), mode="constant", constant_values=0)

    sequence = sequence.reshape(1, time_steps, 1)
    model = load_model(model)  
    predicted_price = model.predict(sequence)[0][0]
    if scaler == None:
        scaler = MinMaxScaler(feature_range=(0, 1))  
    predicted_price = scaler.inverse_transform(predicted_price.reshape(1, -1))
    print(f"The Petrol Price for the year {target_date} is: USD{predicted_price} ")
    # return predicted_price #[0][0] 
    

