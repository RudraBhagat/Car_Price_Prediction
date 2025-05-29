from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

# Configure MySQL connection (update with your credentials)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:password123@localhost/Car_Price_DB'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the table structure to match your MySQL `car_price` table
class CarPrediction(db.Model):
    __tablename__ = 'car_price'
    id = db.Column(db.Integer, primary_key=True)
    brand = db.Column(db.String(50))
    model = db.Column(db.String(50))
    year = db.Column(db.Integer)
    fuel_type = db.Column(db.String(50))
    transmission = db.Column(db.String(50))
    kms_driven = db.Column(db.Integer)
    doors = db.Column(db.Integer)
    owner_count = db.Column(db.Integer)
    predicted_price = db.Column(db.Float)

# Load dataset and model
df = pd.read_csv('car_price_dataset.csv')
model = pickle.load(open('random_forest_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

# Dropdown values
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
    # Get input from form
    brand = request.form['brand']
    model_name = request.form['model']
    year = int(request.form['year'])
    fuel_type = request.form['fuel_type']
    transmission = request.form['transmission']
    kms_driven = int(request.form['kms_driven'])
    doors = int(request.form['doors'])
    owner_count = int(request.form['owner_count'])

    # Prepare input for model
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

    # One-hot encode to match training columns
    combined_df = pd.concat([df, input_data], ignore_index=True)
    combined_df = pd.get_dummies(combined_df, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'])
    input_data_encoded = combined_df.iloc[-1:, :]
    input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict price
    prediction = model.predict(input_data_encoded)
    predicted_price = round(prediction[0], 2)

    # Store input and result in MySQL
    new_entry = CarPrediction(
        brand=brand,
        model=model_name,
        year=year,
        fuel_type=fuel_type,
        transmission=transmission,
        kms_driven=kms_driven,
        doors=doors,
        owner_count=owner_count,
        predicted_price=predicted_price
    )
    db.session.add(new_entry)
    db.session.commit()

    return render_template('index.html',
                           brands=brands,
                           fuel_types=fuel_types,
                           transmissions=transmissions,
                           prediction_text=f"Predicted Price: ${predicted_price}")

if __name__ == '__main__':
    app.run(debug=True)
