import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
st.write("""
        # Heart Attack Analysis & Prediction
        ## Made by Qadir Shahbaz""")

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
    age = st.sidebar.slider("age", 20.0,30.0,40.0,50.0,60.0,70.0,80.0, step=0.5)
    sex = st.sidebar.slider("sex",0.0, 1.0)
    cp = st.sidebar.slider("Chest pain", 1.0,2.0,3.0)
    data = {"age" : age,
            "sex" : sex,
            "cp" : cp}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()
    