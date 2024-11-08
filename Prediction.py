import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/RukmalRt/ckd_predict_1/main/kidney_disease_cleaned2.csv')

# Train/test split
train, test = train_test_split(df, test_size=0.2, random_state=42)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# Feature and target variables
X_train = train.drop('classification', axis=1)
y_train = train['classification']
X_test = test.drop('classification', axis=1)
y_test = test['classification']

# Preprocess data: Separate categorical and numerical features
cat = []
num = []

for col in X_train.columns:
    if X_train[col].dtype == 'object':
        cat.append(col)
    else:
        num.append(col)

# Encode categorical variables
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoded_data = enc.fit_transform(X_train[cat])
encoded_df = pd.DataFrame(encoded_data, columns=cat)
X_train = pd.concat([X_train[num], encoded_df], axis=1)

encoded_data2 = enc.transform(X_test[cat])
encoded_df2 = pd.DataFrame(encoded_data2, columns=cat)
X_test = pd.concat([X_test[num], encoded_df2], axis=1)

# Store column names before scaling
cols = X_train.columns

# Normalize data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=cols)

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=cols)

# Label encode target variable
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Define RandomForest model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Streamlit interface
st.title("Kidney Disease Prediction")
st.write("### Enter Patient Data for Prediction")

# Input fields
age = st.number_input("Age", min_value=0, max_value=200)
bp = st.number_input("Blood Pressure", min_value=0, max_value=500)
sg = st.number_input("Specific Gravity", min_value=1.00, max_value=2.00)
al = st.number_input("Albumin", min_value=0, max_value=10)
su = st.number_input("Sugar Level", min_value=0, max_value=10)
bgr = st.number_input("Blood Glucose", min_value=0, max_value=2000)
bu = st.number_input("Blood Urea", min_value=0, max_value=2000)
sc = st.number_input("Serum Creatine", min_value=0.0, max_value=1000.0)
sod = st.number_input("Sodium", min_value=0, max_value=2000)
pot = st.number_input("Potassium", min_value=0.0, max_value=1000.0)
hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=200.0)
pcv = st.number_input("Packed Cell Volume", min_value=0.0, max_value=100.0)
wc = st.number_input("White Blood Cell Count", min_value=0.0, max_value=100000.0)
rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=100.0)

# Categorical fields as dropdowns
rbc = st.selectbox("Red Blood Cell Clusters", ['normal', 'abnormal'])
pc = st.selectbox("Pus Cell", ['normal', 'abnormal'])
pcc = st.selectbox("Pus Cell Clumps", ['normal', 'abnormal'])
ba = st.selectbox("Bacteria", ['notpresent', 'present'])
htn = st.selectbox("Hypertension", ['yes', 'no'])
dm = st.selectbox("Diabetes Mellitus", ['yes', 'no'])
cad = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
appet = st.selectbox("Appetite", ['good', 'poor'])
pe = st.selectbox("Pedal Edema", ['yes', 'no'])
ane = st.selectbox("Anemia", ['yes', 'no'])

# Define the prediction function
def predict(model):
    data = [age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane]
    data = np.array(data).reshape(1, -1)
    df_input = pd.DataFrame(data, columns=X_train.columns)

    # Preprocess the input data (similar to training data)
    encoded_input = enc.transform(df_input[cat])
    encoded_df_input = pd.DataFrame(encoded_input, columns=cat)
    df_input = pd.concat([df_input[num], encoded_df_input], axis=1)
    df_input = scaler.transform(df_input)
    df_input = pd.DataFrame(df_input, columns=cols)

    # Make prediction
    prediction = model.predict(df_input)
    prediction_label = le.inverse_transform(prediction)[0]

    return prediction_label

# Button to trigger prediction
if st.button("Predict"):
    prediction_result = predict(model)
    st.write(f"The prediction result is: {prediction_result}")
