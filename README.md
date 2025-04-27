# Customer Churn Prediction

This project implements a machine learning model to predict customer churn using artificial neural networks (ANN). The model is deployed through a user-friendly web interface built with Streamlit.

## Project Overview

The Customer Churn Prediction system helps businesses identify customers who are likely to leave their service. It uses various customer attributes such as credit score, age, tenure, balance, and other behavioral factors to predict the probability of churn.

## Features

- Interactive web interface for real-time predictions
- Support for multiple geographical regions
- Comprehensive customer attribute analysis
- Real-time probability scoring
- Pre-trained machine learning model
- Data preprocessing pipeline

## Tech Stack

- Python 3.x
- TensorFlow/Keras for the neural network
- Streamlit for the web interface
- Pandas for data manipulation
- Scikit-learn for data preprocessing
- Pickle for model serialization

## Project Structure

```
├── app.py                    # Streamlit web application
├── experiments.ipynb         # Model development notebook
├── prediction.ipynb         # Prediction experiments
├── model.h5                 # Trained neural network model
├── Churn_Modelling.csv      # Dataset
├── requirements.txt         # Project dependencies
├── scaler.pkl              # Standard scaler for numerical features
├── label_encoder_gender.pkl # Encoder for gender feature
├── onehot_encoder_geo.pkl   # Encoder for geography feature
└── logs/                    # TensorBoard logs
```

## Installation

1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```
streamlit run app.py
```

The web interface will open in your default browser, where you can:
1. Input customer information
2. Get real-time churn predictions
3. View prediction probabilities

## Model Information

The model is built using a deep neural network with:
- Multiple dense layers
- Dropout for regularization
- Early stopping to prevent overfitting
- Binary classification output

## Data Preprocessing

The system handles:
- Categorical encoding for geography and gender
- Feature scaling for numerical attributes
- Missing value handling
- Feature engineering

## Contributing

Feel free to open issues and pull requests to improve the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
