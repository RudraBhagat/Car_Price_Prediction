import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pymysql

# --- Streamlit UI Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Car Price Predictor", layout="centered")

# --- Database Configuration (Update with your MySQL credentials) ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password123',
    'database': 'Car_Price_DB'
}

# --- Function to establish database connection ---
@st.cache_resource
def get_db_connection():
    """Establishes and returns a database connection.
    
    NOTE: If you encounter an error like "'cryptography' package is required for sha256_password...",
    you need to install the 'cryptography' package: pip install cryptography
    This is often required for newer MySQL authentication methods.
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

# --- Load Dataset and Model ---
# Use st.cache_data to cache the loading of heavy resources
@st.cache_data
def load_data():
    """Loads the car price dataset."""
    try:
        df = pd.read_csv('car_price_dataset.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'car_price_dataset.csv' not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if essential file is missing
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

@st.cache_resource
def load_model_and_columns():
    """Loads the trained model and model columns."""
    try:
        model = pickle.load(open('random_forest_model.pkl', 'rb'))
        model_columns = pickle.load(open('model_columns.pkl', 'rb'))
        return model, model_columns
    except FileNotFoundError:
        st.error("Error: 'random_forest_model.pkl' or 'model_columns.pkl' not found. Please ensure they are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or columns: {e}")
        st.stop()

df = load_data()
model, model_columns = load_model_and_columns()

# --- Prepare Dropdown Values ---
# Ensure unique values and sort them for consistent display
brands = sorted(df['Brand'].unique())
fuel_types = sorted(df['Fuel_Type'].unique())
transmissions = sorted(df['Transmission'].unique())

# --- Streamlit UI ---
st.title("🚗 Car Price Predictor")
st.markdown("Enter the details of the car to get a price prediction.")

# Create input columns for better layout
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brands, help="Select the car brand.")
    # Dynamically filter models based on selected brand
    if brand:
        filtered_models = df[df['Brand'].str.strip().str.lower() == brand.strip().lower()]['Model'].unique()
        models_list = sorted(filtered_models)
    else:
        models_list = [] # No brand selected, no models to show

    model_name = st.selectbox("Model", models_list, help="Select the car model.")
    year = st.number_input("Year", min_value=1990, max_value=2024, value=2015, step=1, help="Manufacturing year of the car.")
    fuel_type = st.selectbox("Fuel Type", fuel_types, help="Type of fuel the car uses.")

with col2:
    transmission = st.selectbox("Transmission", transmissions, help="Transmission type (Manual/Automatic).")
    kms_driven = st.number_input("Kms Driven", min_value=0, max_value=500000, value=50000, step=1000, help="Total kilometers driven by the car.")
    doors = st.number_input("Doors", min_value=2, max_value=5, value=4, step=1, help="Number of doors in the car.")
    owner_count = st.number_input("Owner Count", min_value=1, max_value=10, value=1, step=1, help="Number of previous owners.")

# --- Prediction Button ---
st.markdown("---") # Separator for better visual
if st.button("Predict Price", help="Click to get the predicted price of the car."):
    if not brand or not model_name:
        st.warning("Please select both Brand and Model.")
    else:
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

        try:
            # Combine with original df for consistent one-hot encoding
            # This ensures all possible columns are present after encoding
            combined_df = pd.concat([df.drop(columns=['Price']), input_data], ignore_index=True)
            combined_df = pd.get_dummies(combined_df, columns=['Brand', 'Model', 'Fuel_Type', 'Transmission'])

            # Extract the encoded input data (the last row)
            input_data_encoded = combined_df.iloc[-1:, :]

            # Reindex to match the model's training columns, filling missing with 0
            input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

            # Predict price
            prediction = model.predict(input_data_encoded)
            predicted_price = round(prediction[0], 2)

            st.success(f"**Predicted Price: ${predicted_price:,.2f}**")

            # --- Store input and result in MySQL ---
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    insert_query = """
                    INSERT INTO car_price (brand, model, year, fuel_type, transmission, kms_driven, doors, owner_count, predicted_price)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (
                        brand, model_name, year, fuel_type, transmission,
                        kms_driven, doors, owner_count, predicted_price
                    ))
                    conn.commit()
                    st.info("Prediction details saved to database.")
                except pymysql.Error as db_err:
                    st.error(f"Error saving to database: {db_err}")
                finally:
                    cursor.close()
                    conn.close()
            else:
                st.warning("Could not save to database due to connection error.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check your input values and try again.")

