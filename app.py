import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle

## load the model
model = tf.keras.models.load_model('model.h5')

## load the scaler
## load the trained model , scaler pickle, one hot encoder pickle

#load the encoder and scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    label_encoder_geo = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

##streamlit app
st.title("Customer Churn Prediction")
#user input
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])  # Use categories_ to get the list of categories
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', 0.0, 100000.0, 50000.0)
credit_score = st.number_input('Credit Score', 0, 850, 700)
estimated_salary = st.number_input('Estimated Salary', 0.0, 200000.0, 50000.0)
tenure = st.number_input('Tenure', 0, 10, 5)
num_of_products = st.number_input('Number of Products', 1, 4, 2)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])


## prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender' : [label_encoder_gender.transform([gender])[0]],
    'Age' : [age],
    'Tenure' : [tenure],
    'Balance' : [balance],
    'NumOfProducts' : [num_of_products],
    'HasCrCard' : [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember' : [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary' : [estimated_salary],
    
})

input_data['Geography'] = [geography]  # Add Geography to the input data before encoding
### one hot encoding 'Geography'
geo_enocded = label_encoder_geo.transform([[input_data['Geography'][0]]]).toarray()  # Access the first element for encoding
geo_enocded_df = pd.DataFrame(geo_enocded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

## combine the input data with the one hot encoded data
input_data = pd.concat([input_data.drop(['Geography'], axis=1).reset_index(drop=True), 
                       geo_enocded_df.reset_index(drop=True)], axis=1)

# Ensure columns are in the correct order
expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                   'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 
                   'Geography_France', 'Geography_Germany', 'Geography_Spain']
input_data = input_data[expected_columns]

# scale the input data
input_data_scaled = scaler.transform(input_data)

#predict the churn
prediction = model.predict(input_data_scaled)
prediction_proba = float(prediction[0][0])  # Convert to float for better display

# Display results with Streamlit
st.write('---')  # Add a divider
st.write('## Prediction Results')
st.write(f'Churn Probability: {prediction_proba:.2%}')

if prediction_proba > 0.5:
    st.error("⚠️ The customer is likely to churn!")
else:
    st.success("✅ The customer is unlikely to churn.")




