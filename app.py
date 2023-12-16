import streamlit as st
import numpy as np
import pickle
import json

# Load the model and label encoder
with open("C:\\Users\\ASUS\\Desktop\\BigData\\Final Project\\CODE\\models\\random_forest_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("C:\\Users\\ASUS\\Desktop\\BigData\\Final Project\\CODE\\models\\label_encoder.pkl", 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Define the Streamlit layout
st.title('Attack Prediction App')

# Create a form for the normal input
with st.form('input_form'):
    st.write('Please enter the details:')
    flag_S0 = st.number_input('flag_S0', min_value=0, max_value=1, step=1, format='%d')
    flag_SF = st.number_input('flag_SF', min_value=0, max_value=1, step=1, format='%d')
    src_bytes = st.number_input('src_bytes', min_value=0)
    dst_bytes = st.number_input('dst_bytes', min_value=0)
    logged_in = st.number_input('logged_in', min_value=0, max_value=1, step=1, format='%d')
    count = st.number_input('count', min_value=0)
    serror_rate = st.number_input('serror_rate', min_value=0.0, max_value=1.0, step=0.01)
    srv_serror_rate = st.number_input('srv_serror_rate', min_value=0.0, max_value=1.0, step=0.01)
    dst_host_srv_count = st.number_input('dst_host_srv_count', min_value=0)
    dst_host_same_srv_rate = st.number_input('dst_host_same_srv_rate', min_value=0.0, max_value=1.0, step=0.01)
    dst_host_diff_srv_rate = st.number_input('dst_host_diff_srv_rate', min_value=0.0, max_value=1.0, step=0.01)
    dst_host_same_src_port_rate = st.number_input('dst_host_same_src_port_rate', min_value=0.0, max_value=1.0, step=0.01)
    dst_host_serror_rate = st.number_input('dst_host_serror_rate', min_value=0.0, max_value=1.0, step=0.01)
    dst_host_srv_serror_rate = st.number_input('dst_host_srv_serror_rate', min_value=0.0, max_value=1.0, step=0.01)
    
    submitted = st.form_submit_button('Submit')
    if submitted:
        # Use the model to predict
        input_data = np.array([[flag_S0, flag_SF, src_bytes, dst_bytes, logged_in, count, serror_rate, 
                                srv_serror_rate, dst_host_srv_count, dst_host_same_srv_rate, 
                                dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_serror_rate, 
                                dst_host_srv_serror_rate]])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        attack_label = label_encoder.inverse_transform(prediction)
        
        # Display the result
        st.write(f'Probability of Attack: {prediction_proba.max()}')
        st.write(f'Type of Attack: {attack_label[0]}')

# Create a text area for JSON input
json_input = st.text_area('Or paste JSON here:', '{}', height=400)
if st.button('Predict from JSON'):
    # Parse the JSON input
    try:
        input_data = json.loads(json_input)
        
        # Define the expected input features based on the model's requirements
        input_features = [
            'flag_S0', 'flag_SF', 'src_bytes', 'dst_bytes', 'logged_in',
            'count', 'serror_rate', 'srv_serror_rate', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate'
        ]
        
        # Validate that all features are in the input JSON
        for feature in input_features:
            if feature not in input_data:
                raise ValueError(f"Missing feature in input: {feature}")
        
        # Extract the feature values in the correct order
        input_values = [input_data[feature] for feature in input_features]
        
        # Make prediction
        prediction = model.predict([input_values])
        prediction_proba = model.predict_proba([input_values])
        attack_label = label_encoder.inverse_transform(prediction)
        
        # Display the result
        st.write(f'Probability of Attack: {prediction_proba.max()}')
        st.write(f'Type of Attack: {attack_label[0]}')
    except json.JSONDecodeError:
        st.error('Invalid JSON')
    except ValueError as e:
        st.error(e)