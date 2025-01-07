# import os
# import re
# from collections import Counter

# import joblib
# import pandas as pd
# from flask import Flask, jsonify, render_template, request

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the model, scaler, and feature columns
# model = joblib.load("models/RandomForest.pkl")
# scaler = joblib.load("models/scaler.pkl")
# feature_columns = joblib.load("models/feature_columns.pkl")  # Load feature column names

# # Feature extraction from chemical formula
# def extract_features(formula):
#     elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
#     element_counts = Counter()
#     for (element, count) in elements:
#         count = int(count) if count else 1
#         element_counts[element] += count
#     return element_counts

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     formula = request.form['formula']
#     temperature = float(request.form['temperature'])

#     # Extract features
#     feature_dict = extract_features(formula)
#     feature_df = pd.DataFrame([feature_dict]).fillna(0)

#     # Add temperature to features
#     feature_df['temperature(K)'] = temperature

#     # Ensure all required columns are present
#     for col in feature_columns:
#         if col not in feature_df:
#             feature_df[col] = 0
#     feature_df = feature_df[feature_columns]  # Align order with training

#     # Scale the features
#     scaled_features = scaler.transform(feature_df)

#     # Predict
#     prediction = model.predict(scaled_features)
#     results = {
#         "Seebeck Coefficient (μV/K)": prediction[0][0],
#         "Electrical Conductivity (S/m)": prediction[0][1],
#         "Thermal Conductivity (W/mK)": prediction[0][2],
#         "Power Factor (W/mK²)": prediction[0][3],
#         "ZT": prediction[0][4],
#     }
#     return jsonify(results)

# @app.route('/predict_file', methods=['POST'])
# def predict_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     # Save the uploaded file
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Load the file into a DataFrame
#     try:
#         data = pd.read_excel(filepath)
#     except Exception as e:
#         return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

#     # Validate required columns
#     if 'Formula' not in data.columns or 'temperature(K)' not in data.columns:
#         return jsonify({"error": "File must contain 'Formula' and 'temperature(K)' columns"}), 400

#     # Extract features
#     feature_dicts = data['Formula'].apply(extract_features)
#     feature_df = pd.DataFrame(list(feature_dicts)).fillna(0)
#     feature_df['temperature(K)'] = data['temperature(K)']

#     # Ensure all required columns are present
#     for col in feature_columns:
#         if col not in feature_df:
#             feature_df[col] = 0
#     feature_df = feature_df[feature_columns]  # Align order with training

#     # Scale the features
#     scaled_features = scaler.transform(feature_df)

#     # Predict
#     predictions = model.predict(scaled_features)
#     # results_df = pd.DataFrame(predictions, columns=["Seebeck Coefficient (μV/K)", "Electrical Conductivity (S/m)", "Thermal Conductivity (W/mK)", "Power Factor (W/mK²)",
#     #     "ZT"
#     # ])
#     results_df = pd.DataFrame(predictions, columns=["Seebeck Coefficient (μV/K)", "Electrical Conductivity (S/m)", "Thermal Conductivity (W/mK)", "Power Factor (W/mK²)", "ZT"])
#     output_df = pd.concat([data, results_df], axis=1)

#     # Save results to a new Excel file
#     output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "predictions.xlsx")
#     output_df.to_excel(output_filepath, index=False)

#     return jsonify({"message": "Predictions completed", "file": output_filepath})

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8099)

# import os
# import re
# from collections import Counter

# import joblib
# import pandas as pd
# from flask import Flask, jsonify, render_template, request, send_from_directory

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the model, scaler, and feature columns
# model = joblib.load("models/RandomForest.pkl")
# scaler = joblib.load("models/scaler.pkl")
# feature_columns = joblib.load("models/feature_columns.pkl")  # Load feature column names

# # Feature extraction from chemical formula
# def extract_features(formula):
#     elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
#     element_counts = Counter()
#     for (element, count) in elements:
#         count = int(count) if count else 1
#         element_counts[element] += count
#     return element_counts

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict_file', methods=['POST'])
# def predict_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     # Save the uploaded file
#     filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(filepath)

#     # Load the file into a DataFrame
#     try:
#         data = pd.read_excel(filepath)
#     except Exception as e:
#         return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

#     # Validate required columns
#     if 'Formula' not in data.columns or 'temperature(K)' not in data.columns:
#         return jsonify({"error": "File must contain 'Formula' and 'temperature(K)' columns"}), 400

#     # Extract features
#     feature_dicts = data['Formula'].apply(extract_features)
#     feature_df = pd.DataFrame(list(feature_dicts)).fillna(0)
#     feature_df['temperature(K)'] = data['temperature(K)']

#     # Ensure all required columns are present
#     for col in feature_columns:
#         if col not in feature_df:
#             feature_df[col] = 0
#     feature_df = feature_df[feature_columns]  # Align order with training

#     # Scale the features
#     scaled_features = scaler.transform(feature_df)

#     # Predict
#     predictions = model.predict(scaled_features)
#     results_df = pd.DataFrame(predictions, columns=["Seebeck Coefficient (μV/K)", "Electrical Conductivity (S/m)", "Thermal Conductivity (W/mK)", "Power Factor (W/mK²)", "ZT"])
#     output_df = pd.concat([data, results_df], axis=1)

#     # Save results to a new Excel file
#     output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "predictions.xlsx")
#     output_df.to_excel(output_filepath, index=False)

#     # Provide download links for both files
#     return jsonify({
#         "message": "Predictions completed",
#         "uploaded_file_link": f"/download/{file.filename}",
#         "predictions_file_link": f"/download/predictions.xlsx"
#     })

# @app.route('/download/<filename>', methods=['GET'])
# def download_file(filename):
#     # Serve the file from the uploads directory
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8080)

# import os
# import re
# import zipfile
# from collections import Counter

# import joblib
# import pandas as pd
# from flask import Flask, jsonify, render_template, request, send_file

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load the model, scaler, and feature columns
# model = joblib.load("models/RandomForest.pkl")
# scaler = joblib.load("models/scaler.pkl")
# feature_columns = joblib.load("models/feature_columns.pkl")  # Load feature column names

# # Feature extraction from chemical formula
# def extract_features(formula):
#     elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
#     element_counts = Counter()
#     for (element, count) in elements:
#         count = int(count) if count else 1
#         element_counts[element] += count
#     return element_counts

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     formula = request.form['formula']
#     temperature = float(request.form['temperature'])

#     # Extract features
#     feature_dict = extract_features(formula)
#     feature_df = pd.DataFrame([feature_dict]).fillna(0)

#     # Add temperature to features
#     feature_df['temperature(K)'] = temperature

#     # Ensure all required columns are present
#     for col in feature_columns:
#         if col not in feature_df:
#             feature_df[col] = 0
#     feature_df = feature_df[feature_columns]  # Align order with training

#     # Scale the features
#     scaled_features = scaler.transform(feature_df)

#     # Predict
#     prediction = model.predict(scaled_features)
    
#     results = {
#         "Seebeck Coefficient (μV/K)": prediction[0][0],
#         "Electrical Conductivity (S/m)": prediction[0][1],
#         "Thermal Conductivity (W/mK)": prediction[0][2],
#         "Power Factor (W/mK²)": prediction[0][3],
#         "ZT": prediction[0][4],
#     }
#     return jsonify(results)

# @app.route('/predict_file', methods=['POST'])
# def predict_file():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     # Save the uploaded file
#     uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(uploaded_filepath)

#     # Load the file into a DataFrame
#     try:
#         data = pd.read_excel(uploaded_filepath)
#     except Exception as e:
#         return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

#     # Validate required columns
#     if 'Formula' not in data.columns or 'temperature(K)' not in data.columns:
#         return jsonify({"error": "File must contain 'Formula' and 'temperature(K)' columns"}), 400

#     # Extract features
#     feature_dicts = data['Formula'].apply(extract_features)
#     feature_df = pd.DataFrame(list(feature_dicts)).fillna(0)
#     feature_df['temperature(K)'] = data['temperature(K)']

#     # Ensure all required columns are present
#     for col in feature_columns:
#         if col not in feature_df:
#             feature_df[col] = 0
#     feature_df = feature_df[feature_columns]  # Align order with training

#     # Scale the features
#     scaled_features = scaler.transform(feature_df)

#     # Predict
#     predictions = model.predict(scaled_features)
#     results_df = pd.DataFrame(predictions, columns=[
#         "Seebeck Coefficient (\u03bcV/K)",
#         "Electrical Conductivity (S/m)",
#         "Thermal Conductivity (W/mK)",
#         "Power Factor (W/mK\u00b2)",
#         "ZT"
#     ])
#     output_df = pd.concat([data, results_df], axis=1)

#     # Save results to a new Excel file
#     predictions_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "predictions.xlsx")
#     output_df.to_excel(predictions_filepath, index=False)

#     # Create a zip file containing both files
#     zip_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "results.zip")
#     with zipfile.ZipFile(zip_filepath, 'w') as zipf:
#         zipf.write(uploaded_filepath, os.path.basename(uploaded_filepath))
#         zipf.write(predictions_filepath, os.path.basename(predictions_filepath))

#     return send_file(zip_filepath, mimetype='application/zip', as_attachment=True, download_name="results.zip")

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=8080)


import os
import re
import zipfile
from collections import Counter

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model, scaler, and feature columns
model = joblib.load("models/RandomForest.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")  # Load feature column names


# Feature extraction from chemical formula
def extract_features(formula):
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    element_counts = Counter()
    for (element, count) in elements:
        count = int(count) if count else 1
        element_counts[element] += count
    return element_counts


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    formula = request.form['formula']
    temperature = float(request.form['temperature'])

    # Extract features
    feature_dict = extract_features(formula)
    feature_df = pd.DataFrame([feature_dict]).fillna(0)

    # Add temperature to features
    feature_df['temperature(K)'] = temperature

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in feature_df:
            feature_df[col] = 0
    feature_df = feature_df[feature_columns]  # Align order with training

    # Scale the features
    scaled_features = scaler.transform(feature_df)

    # Predict
    prediction = model.predict(scaled_features)

    results = {
        "Seebeck Coefficient (μV/K)": prediction[0][0],
        "Electrical Conductivity (S/m)": prediction[0][1],
        "Thermal Conductivity (W/mK)": prediction[0][2],
        "Power Factor (W/mK²)": prediction[0][3],
        "ZT": prediction[0][4],
    }

    return jsonify(results)


@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(uploaded_filepath)

    # Load the file into a DataFrame
    try:
        data = pd.read_excel(uploaded_filepath)
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

    # Validate required columns
    if 'Formula' not in data.columns or 'temperature(K)' not in data.columns:
        return jsonify({"error": "File must contain 'Formula' and 'temperature(K)' columns"}), 400

    # Extract features
    feature_dicts = data['Formula'].apply(extract_features)
    feature_df = pd.DataFrame(list(feature_dicts)).fillna(0)
    feature_df['temperature(K)'] = data['temperature(K)']

    # Ensure all required columns are present
    for col in feature_columns:
        if col not in feature_df:
            feature_df[col] = 0
    feature_df = feature_df[feature_columns]  # Align order with training

    # Scale the features
    scaled_features = scaler.transform(feature_df)

    # Predict
    predictions = model.predict(scaled_features)

    results_df = pd.DataFrame(predictions, columns=[
        "Seebeck Coefficient (\u03bcV/K)",
        "Electrical Conductivity (S/m)",
        "Thermal Conductivity (W/mK)",
        "Power Factor (W/mK\u00b2)",
        "ZT"
    ])

    output_df = pd.concat([data, results_df], axis=1)

    # Save results to a new Excel file
    predictions_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "predictions.xlsx")
    output_df.to_excel(predictions_filepath, index=False)

    # Create a zip file containing both files
    zip_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "results.zip")
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        zipf.write(uploaded_filepath, os.path.basename(uploaded_filepath))
        zipf.write(predictions_filepath, os.path.basename(predictions_filepath))

    return send_file(zip_filepath, mimetype='application/zip', as_attachment=True, download_name="results.zip")


@app.route('/uploads/predictions.xlsx')
def download_predictions():
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], 'predictions.xlsx', as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)