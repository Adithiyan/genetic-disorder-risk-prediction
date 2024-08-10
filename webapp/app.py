import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

try:
    with open('logistic_model.pkl', 'rb') as f:
        logistic_classifier = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('unique_categories.pkl', 'rb') as f:
        unique_categories = pickle.load(f)
except FileNotFoundError:
    print("Model or encoders not found")
    logistic_classifier = None
    label_encoders = None
    unique_categories = None

def predict(input_data):
    if logistic_classifier is None or label_encoders is None or unique_categories is None:
        return "Model or encoders not loaded properly"
    
    # Define categorical columns
    categorical_columns = ['Type', 'Genetic_disease', 'Gene', 'Variation', 'Consanguinity', 'Mother_affected', 'Mother_Carrier', 'Father_affected', 'Father_Carrier', '1st_level', '2nd_level', 'G1', 'G2', 'G3', 'G4', 'G5']
    
    # Encode categorical features
    encoded_data = []
    for i, col in enumerate(categorical_columns):
        value = input_data[i]
        if value:
            value = str(value)  # Ensure the value is a string
            if value in unique_categories[col]:
                encoded_value = label_encoders[col].transform([value])[0]
            else:
                # Handle unknown category values
                st.write(f"Value {value} for column {col} not found in label encoder")
                encoded_value = -1  # Use a value to indicate unknown category
        else:
            # Handle missing or empty input data
            st.write(f"Value for column {col} is empty or missing")
            encoded_value = -1
        encoded_data.append(encoded_value)
    
    input_df = pd.DataFrame([input_data], columns=categorical_columns)
    st.write("Original Input:", input_df)

    encoded_df = pd.DataFrame([encoded_data], columns=categorical_columns)
    st.write("Encoded Input:", encoded_df)
    
    # Make prediction
    prediction = logistic_classifier.predict(encoded_df)
    return prediction[0]

input_data = ['Type_value', 'Genetic_disease_value', 'Gene_value', 'Variation_value', 'Consanguinity_value', 'Mother_affected_value', 'Mother_Carrier_value', 'Father_affected_value', 'Father_Carrier_value', '1st_level_value', '2nd_level_value', 'G1_value', 'G2_value', 'G3_value', 'G4_value', 'G5_value']

def main():
    st.title("Genetic Disorder Risk Prediction")

    input_data = []
    options_type = ['Autosomal Recessive', 'Autosomal Dominant', 'X-linked Recessive', 'X-linked Dominant']
    input_data.append(st.radio("Type", options_type))

    input_data.append(st.text_input("Genetic_disease"))
    input_data.append(st.text_input("Gene"))

    options_variation = ['Homozygous', 'Compound Heterozygous', 'Heterozygous', 'Hemizygous']
    input_data.append(st.radio("Variation", options_variation))

    options1 = ['Yes', 'No', 'NIL']
    input_data.append(st.radio("Consanguinity", options1))
    input_data.append(st.radio("Mother_affected", options1))
    input_data.append(st.radio("Mother_Carrier", options1))
    input_data.append(st.radio("Father_affected", options1))
    input_data.append(st.radio("Father_Carrier", options1))
    input_data.append(st.radio("1st_level", options1))
    input_data.append(st.radio("2nd_level", options1))

    options2 = ['Yes', 'No', 'MTP', 'Abortion', 'NIL']
    input_data.append(st.radio("G1", options2))
    input_data.append(st.radio("G2", options2))

    options3 = ['NIL', 'Yes', 'No', 'MTP', 'Abortion']
    input_data.append(st.radio("G3", options3))
    input_data.append(st.radio("G4", options3))
    input_data.append(st.radio("G5", options3))


    
    print(input_data)
    # Make prediction
    if st.button("Predict"):
        prediction = predict(input_data)
        

        # Map bins to percentages
        bin_to_percentage = {
            1: "4%",
            2: "25%",
            3: "50%"
        }
        risk_percentage = bin_to_percentage.get(prediction, "Unknown")
        
        st.write(f"Predicted Risk Percentage: {risk_percentage}")
        st.write(f"Prediction Category: {prediction}")
if __name__ == "__main__":
    main()
