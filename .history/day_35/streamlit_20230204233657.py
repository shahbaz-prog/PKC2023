import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
st.write("""
        # Heart Attack Analysis & Prediction
        ## Made by Qadir Shahbaz""")

st.sidebar.header("User Input Parameters")

st.write(""" ## Heart attack Analysis dataset's keys definition\
         **Age** : Age of the patient

**Sex** : Sex of the patient

**cp** : Chest Pain type

    Value 0: Typical angina

    Value 1: Atypical angina

    Value 2: Non-anginal pain

    Value 3: Asymptomatic

**trtbps** : Blood pressure after receiving treatment (in mm Hg)

**chol**: Cholesterol in mg/dl fetched via BMI sensor

**fbs**: (fasting blood sugar > 120 mg/dl)

    1 = true

    0 = false

**rest_ecg**: Resting electrocardiographic results
    Value 0: normal

    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

**thalach**: Maximum heart rate achieved

**exang**: Exercise induced angina(discomfort du)

    1 = yes

    0 = no

**old peak**: ST depression induced by exercise relative to rest

**slp**: The slope of the peak exercise ST segment

    0 = Unsloping

    1 = flat

    2 = downsloping

**caa**: Number of major vessels (0-3)

**thall** : Thalassemia

    0 = null

    1 = fixed defect

    2 = normal

    3 = reversable defect

**output**: diagnosis of heart disease (angiographic disease status)

    0: < 50% diameter narrowing. less chance of heart disease

    1: > 50% diameter narrowing. more chance of heart disease""")