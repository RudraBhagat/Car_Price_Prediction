from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load dataset and model
df = pd.read_csv('car_price_dataset.csv')
model = pickle.load(open('random_forest_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Get unique values for dropdowns
brands = sorted(df['Brand'].unique())
fuel_types = sorted(df['Fuel_Type'].unique())
transmissions = sorted(df['Transmission'].unique())

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', brands=brands, fuel_types=fuel_types, transmissions=transmissions)

@app.route('/get_models', methods=['POST'])
def get_models():
    brand_name = request.json['brand'].strip()
    filtered_models = df[df['Brand'].str.strip().str.lower() == brand_name.lower()]['Model'].unique()
    models_list = sorted(filtered_models)
    return jsonify(models=models_list)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    brand = request.form['brand']
    model_name = request.form['model']
    year = int(request.form['year'])
    fuel_type = request.form['fuel_type']
    transmission = request.form['transmission']
    kms_driven = int(request.form['kms_driven'])
    doors = int(request.form['doors'])
    owner_count = int(request.form['owner_count'])
    
    # Prepare the input DataFrame
    input_data = pd.DataFrame({
        'Brand': [brand],
        'Model': [model_name],
        'Year': [year],
        'Fuel_Type': [fuel_type],
        'Transmission': [transmission],
        'Kms_Driven': [kms_driven],
        'Doors': [doors],
        'Owner_Count': [owner_count]
    })

    # Encode categorical columns
    combined_df = pd.concat([df, input_data], ignore_index=True)
    combined_df = pd.get_dummies(combined_df, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'])
    
    # Ensure input data has same columns as training data
    input_data_encoded = combined_df.iloc[-1:, :]
    input_data_encoded = input_data_encoded.reindex(columns=model_columns)
    input_data_encoded = input_data_encoded.fillna(0)
    
    # Make prediction
    prediction = model.predict(input_data_encoded)
    predicted_price = round(prediction[0], 2)
    
    return render_template('index.html', brands=brands, fuel_types=fuel_types, 
                           transmissions=transmissions, prediction_text=f"Predicted Price: ${predicted_price}")

if __name__ == '__main__':
    app.run(debug=True)
