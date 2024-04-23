import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle

st.write("""
# Disease Prediction App

This app predicts the **Disease Classification**!
""")
st.write('---')

# Loads the model
with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)  # Load your trained model

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

import pandas as pd
import streamlit as st

def user_input_features():
    Glucose = st.sidebar.slider('Glucose', X.Glucose.min(), X.Glucose.max(), X.Glucose.mean())
    Cholesterol = st.sidebar.slider('Cholesterol', X.Cholesterol.min(), X.Cholesterol.max(), X.Cholesterol.mean())
    Hemoglobin = st.sidebar.slider('Hemoglobin', X.Hemoglobin.min(), X.Hemoglobin.max(), X.Hemoglobin.mean())
    Platelets = st.sidebar.slider('Platelets', X.Platelets.min(), X.Platelets.max(), X.Platelets.mean())
    White_Blood_Cells = st.sidebar.slider('White Blood Cells', X['White Blood Cells'].min(), X['White Blood Cells'].max(), X['White Blood Cells'].mean())
    Red_Blood_Cells = st.sidebar.slider('Red Blood Cells', X['Red Blood Cells'].min(), X['Red Blood Cells'].max(), X['Red Blood Cells'].mean())
    Hematocrit = st.sidebar.slider('Hematocrit', X.Hematocrit.min(), X.Hematocrit.max(), X.Hematocrit.mean())
    Mean_Corpuscular_Volume = st.sidebar.slider('Mean Corpuscular Volume', X['Mean Corpuscular Volume'].min(), X['Mean Corpuscular Volume'].max(), X['Mean Corpuscular Volume'].mean())
    Mean_Corpuscular_Hemoglobin = st.sidebar.slider('Mean Corpuscular Hemoglobin', X['Mean Corpuscular Hemoglobin'].min(), X['Mean Corpuscular Hemoglobin'].max(), X['Mean Corpuscular Hemoglobin'].mean())
    Mean_Corpuscular_Hemoglobin_Concentration = st.sidebar.slider('Mean Corpuscular Hemoglobin Concentration', X['Mean Corpuscular Hemoglobin Concentration'].min(), X['Mean Corpuscular Hemoglobin Concentration'].max(), X['Mean Corpuscular Hemoglobin Concentration'].mean())
    Insulin = st.sidebar.slider('Insulin', X.Insulin.min(), X.Insulin.max(), X.Insulin.mean())
    BMI = st.sidebar.slider('BMI', X.BMI.min(), X.BMI.max(), X.BMI.mean())
    Systolic_Blood_Pressure = st.sidebar.slider('Systolic Blood Pressure', X['Systolic Blood Pressure'].min(), X['Systolic Blood Pressure'].max(), X['Systolic Blood Pressure'].mean())
    Diastolic_Blood_Pressure = st.sidebar.slider('Diastolic Blood Pressure', X['Diastolic Blood Pressure'].min(), X['Diastolic Blood Pressure'].max(), X['Diastolic Blood Pressure'].mean())
    Triglycerides = st.sidebar.slider('Triglycerides', X.Triglycerides.min(), X.Triglycerides.max(), X.Triglycerides.mean())
    HbA1c = st.sidebar.slider('HbA1c', X.HbA1c.min(), X.HbA1c.max(), X.HbA1c.mean())
    LDL_Cholesterol = st.sidebar.slider('LDL Cholesterol', X['LDL Cholesterol'].min(), X['LDL Cholesterol'].max(), X['LDL Cholesterol'].mean())
    ALT = st.sidebar.slider('ALT', X.ALT.min(), X.ALT.max(), X.ALT.mean())
    AST = st.sidebar.slider('AST', X.AST.min(), X.AST.max(), X.AST.mean())
    Heart_Rate = st.sidebar.slider('Heart Rate', X['Heart Rate'].min(), X['Heart Rate'].max(), X['Heart Rate'].mean())
    Creatinine = st.sidebar.slider('Creatinine', X.Creatinine.min(), X.Creatinine.max(), X.Creatinine.mean())
    Troponin = st.sidebar.slider('Troponin', X.Troponin.min(), X.Troponin.max(), X.Troponin.mean())
    C_reactive_Protein = st.sidebar.slider('C-reactive Protein', X['C-reactive Protein'].min(), X['C-reactive Protein'].max(), X['C-reactive Protein'].mean())

    data = {
        'Glucose': Glucose,
        'Cholesterol': Cholesterol,
        'Hemoglobin': Hemoglobin,
        'Platelets': Platelets,
        'White Blood Cells': White_Blood_Cells,
        'Red Blood Cells': Red_Blood_Cells,
        'Hematocrit': Hematocrit,
        'Mean Corpuscular Volume': Mean_Corpuscular_Volume,
        'Mean Corpuscular Hemoglobin': Mean_Corpuscular_Hemoglobin,
        'Mean Corpuscular Hemoglobin Concentration': Mean_Corpuscular_Hemoglobin_Concentration,
        'Insulin': Insulin,
        'BMI': BMI,
        'Systolic Blood Pressure': Systolic_Blood_Pressure,
        'Diastolic Blood Pressure': Diastolic_Blood_Pressure,
        'Triglycerides': Triglycerides,
        'HbA1c': HbA1c,
        'LDL Cholesterol': LDL_Cholesterol,
        'HDL Cholesterol': HDL_Cholesterol,
        'ALT': ALT,
        'AST': AST,
        'Heart Rate': Heart_Rate,
        'Creatinine': Creatinine,
        'Troponin': Troponin,
        'C-reactive Protein': C_reactive_Protein
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()


# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Apply Model to Make Prediction
prediction = rf_model.predict(df)

st.header('Prediction of Disease')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')