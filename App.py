import pandas as pd
from sklearn.model_selection import train_test_split
pip install scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Step 1: Data Preprocessing
def preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Handle missing values in the RiskLevel column
    data['RiskLevel'] = data['RiskLevel'].fillna(data['RiskLevel'].mode()[0])
    
    # Standardize RiskLevel values (strip spaces, capitalize)
    data['RiskLevel'] = data['RiskLevel'].str.strip().str.capitalize()
    
    # Map RiskLevel to integers
    risk_mapping = {'Low risk': 0, 'Mid risk': 1, 'High risk': 2}
    data['RiskLevel'] = data['RiskLevel'].map(risk_mapping)
    
    # Handle any remaining unmapped or missing values by filling with 'Low'
    if data['RiskLevel'].isnull().sum() > 0:
        data['RiskLevel'] = data['RiskLevel'].fillna(0)  # Default to 'Low'
    
    # Separate features (X) and target (y)
    X = data[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
    y = data['RiskLevel']
    
    return X, y

# Step 2: Machine Learning Model Training
def train_model(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model (optional for debugging)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

# Streamlit App
def main():
    st.title("Maternal Health Risk Predictor")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file):", type="csv")
    
    if uploaded_file is not None:
        # Preprocess the data
        X, y = preprocess_data(uploaded_file)
        
        # Train the model
        model, accuracy = train_model(X, y)
        
        st.success(f"Model trained successfully with an accuracy of {accuracy:.2f}")
        
        st.subheader("Enter patient details:")
        
        # Input fields
        age = st.number_input("Age", min_value=10, max_value=100, value=25, step=1)
        systolic_bp = st.number_input("SystolicBP", min_value=50, max_value=200, value=120, step=1)
        diastolic_bp = st.number_input("DiastolicBP", min_value=30, max_value=150, value=80, step=1)
        bs = st.number_input("Blood Sugar (BS)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
        body_temp = st.number_input("Body Temperature", min_value=35.0, max_value=42.0, value=37.0, step=0.1)
        heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75, step=1)
        
        # Predict button
        if st.button("Predict Risk Level"):
            # Create a dataframe for the input
            user_input = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                                      columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
            
            # Predict risk level
            risk_level_num = model.predict(user_input)[0]
            risk_mapping = {0: "Low Risk", 1: "Mid Risk", 2: "High Risk"}
            risk_level = risk_mapping[risk_level_num]
            
            st.success(f"Predicted Risk Level: {risk_level}")

# Run the app
if __name__ == "__main__":
    main()
