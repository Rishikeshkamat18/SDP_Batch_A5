import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Step 2: Machine Learning Model Training and Testing
def train_and_test_model(X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model (but don't print accuracy)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)  # Accuracy is calculated but not printed
    
    return model

# Step 3: Model Prediction
def predict_risk(model):
    print("\nEnter the following details to predict RiskLevel:")
    age = int(input("Age: "))
    systolic_bp = int(input("SystolicBP: "))
    diastolic_bp = int(input("DiastolicBP: "))
    bs = float(input("BS (Blood Sugar): "))
    body_temp = float(input("BodyTemp: "))
    heart_rate = int(input("HeartRate: "))

    # Create a dataframe for user input
    user_input = pd.DataFrame([[age, systolic_bp, diastolic_bp, bs, body_temp, heart_rate]],
                              columns=['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate'])
    
    # Predict risk level (it will return 0, 1, or 2)
    risk_level_num = model.predict(user_input)[0]
    
    # Print the predicted risk level (0, 1, or 2)
    print(f"Predicted Risk Level: {risk_level_num}")

# Main Function
if _name_ == "_main_":
    # File path to your dataset
    file_path = r"C:\Users\ASUS\Downloads\archive (7)\Maternal Health Risk Data Set.csv"  # Update with your actual file path
    
    # Preprocess the data
    X, y = preprocess_data(file_path)
    
    # Train and test the model
    trained_model = train_and_test_model(X, y)
    
    # Predict risk based on user input
    predict_risk(trained_model)
