from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os

app = Flask(__name__)

# Load model and encoders
model = load_model('models/cnn_model.h5')
protocol_encoder = joblib.load('models/protocol_encoder.pkl')
service_encoder = joblib.load('models/service_encoder.pkl')
flag_encoder = joblib.load('models/flag_encoder.pkl')

# Feature columns used in training
desired_columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    error = None
    predictions = None
    try:
        if 'file' not in request.files:
            raise ValueError("No file part")

        file = request.files['file']
        if file.filename == '':
            raise ValueError("No selected file")

        # Read uploaded CSV (no header)
        df = pd.read_csv(file, header=None)

        # Assign column names (KDD dataset has 42 columns + label + difficulty)
        column_names = desired_columns + ['label', 'difficulty']
        df.columns = column_names

        # Filter only required columns
        df = df[desired_columns]

        # Encode categorical features
        protocol_ohe = protocol_encoder.transform(df[['protocol_type']])
        service_ohe = service_encoder.transform(df[['service']])
        flag_ohe = flag_encoder.transform(df[['flag']])

        # Convert to DataFrames
        protocol_df = pd.DataFrame(protocol_ohe, columns=protocol_encoder.get_feature_names_out(['protocol_type']))
        service_df = pd.DataFrame(service_ohe, columns=service_encoder.get_feature_names_out(['service']))
        flag_df = pd.DataFrame(flag_ohe, columns=flag_encoder.get_feature_names_out(['flag']))

        # Reset index to align
        df = df.reset_index(drop=True)
        protocol_df = protocol_df.reset_index(drop=True)
        service_df = service_df.reset_index(drop=True)
        flag_df = flag_df.reset_index(drop=True)

        # Drop original categorical columns and concatenate
        df = df.drop(['protocol_type', 'service', 'flag'], axis=1)
        df = pd.concat([df, protocol_df, service_df, flag_df], axis=1)

        # Clean and reshape
        df = df.replace('?', 0)
        df = df.astype(float)
        X = np.array(df)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Make predictions
        y_pred = model.predict(X)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Show single output: Attack Detected or Normal Traffic
        if np.any(y_pred_classes != 0):
            predictions = "Attack Detected"
        else:
            predictions = "Normal Traffic"

    except Exception as e:
        error = f"Error processing file: {str(e)}"

    return render_template('index.html', error=error, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
