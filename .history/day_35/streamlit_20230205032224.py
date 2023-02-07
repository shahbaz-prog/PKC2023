import streamlit as st
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly as px
import matplotlib.pyplot as plt
from dataprep.eda import *
from dataprep.datasets import load_dataset
from dataprep.eda import plot, plot_correlation, plot_missing, plot_diff, create_report

st.write("""
        # Heart Attack Analysis & Prediction
        ## Made by Qadir Shahbaz & Team""")

st.sidebar.header("User Input Parameters")

st.write(""" ## Heart attack Analysis dataset's keys definition
         
**1. Age** : Age of the patient

**2. Sex** : Sex of the patient

**3. cp** : Chest Pain type

    Value 0: Typical angina

    Value 1: Atypical angina

    Value 2: Non-anginal pain

    Value 3: Asymptomatic

**4. trtbps** : Blood pressure after receiving treatment (in mm Hg)

**5. chol**: Cholesterol in mg/dl fetched via BMI sensor

**6. fbs**: (Fasting blood sugar > 120 mg/dl)

    1 = true

    0 = false

**7. rest_ecg**: Resting electrocardiographic results
    Value 0: normal

    Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)

    Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria

**8. thalach**: Maximum heart rate achieved

**9.exang**: Exercise induced angina(discomfort du)

    1 = yes

    0 = no

**10. old peak**: ST depression induced by exercise relative to rest

**11. slp**: The slope of the peak exercise ST segment

    0 = Unsloping

    1 = flat

    2 = downsloping

**12. caa**: Number of major vessels (0-3)

**13. thall** : Thalassemia

    0 = null

    1 = fixed defect

    2 = normal

    3 = reversable defect

**14. output**: diagnosis of heart disease (angiographic disease status)

    0: < 50% diameter narrowing. less chance of heart disease

    1: > 50% diameter narrowing. more chance of heart disease""")

def user_input_features():
    min_value = 25
    max_value = 80
    min_value_1 = 90
    max_value_1 = 200
    min_value_2 = 125
    max_value_2 = 565
    min_value_3 = 70
    max_value_3 = 202
    min_value_4 = 0.0
    max_value_4 = 7.0
    min_value_5 = 0
    max_value_5 = 4
    min_value_6 = 0
    max_value_6 = 3
    age = st.sidebar.slider("age", min_value, max_value, value=None, step=None, format=None)
    sex = st.sidebar.slider("sex",0, 1)
    cp = st.sidebar.slider("Chest pain", 1,2,3)
    trtbps = st.sidebar.slider("trtbps", min_value_1, max_value_1, value=None, step=None, format=None)
    chol = st.sidebar.slider("chol", min_value_2, max_value_2, value=None, step=None, format=None)
    fbs = st.sidebar.slider("fbs", 1,2,3)
    restecg = st.sidebar.slider("restecg", 0,1)
    thalach = st.sidebar.slider("thalach", min_value_3, max_value_3, value=None, step=None, format=None)
    exng = st.sidebar.slider("exng", 0,1)
    oldpeak = st.sidebar.slider("oldpeak", min_value_4, max_value_4, value=None, step=None, format=None)
    slp = st.sidebar.slider("slp", 0,1,2)
    caa = st.sidebar.slider("caa",min_value_5, max_value_5, value=None, step=None, format=None)
    thall = st.sidebar.slider("thall",min_value_6, max_value_6, value=None, step=None, format=None )
    
    
    data = {"age" : age,
            "sex" : sex,
            "cp" : cp,
            "trtbps" : trtbps,
            "chol" : chol,
            "fbs" : fbs,
            "restecg" : restecg,
            "thalach" : thalach,
            "exng" : exng,
            "oldpeak" : oldpeak,
            "slp" : slp,
            "caa" : caa,
            "thall" : thall}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

st.subheader("Heart Attack parameters")
st.write(df)

df1 = pd.read_csv("Heart.csv")

st.subheader("Heart Attack dataset")
st.write(df1)

st.title("Plotly to make visualization")

plot(df1)