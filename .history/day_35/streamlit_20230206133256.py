# import libraries


import streamlit as st
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import plotly as px
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
st.write("<b>This text will be bold</b>", unsafe_allow_html=False)



# Insert Pictures
st.image("th.jpg")
st.image("th2.jpg")

# Insert Heading and Subheading
st.write("""
        # Heart Attack Analysis & Prediction
        ## Made by Qadir Shahbaz & Team""")

# Insert sidebar to upload dataset to the streamlit app CSV format
st.markdown("""
            # **01. Exploratory Data Analysis**""")
with st.sidebar.header("Upload your dataset(.csv)"):
    uploaded_file = st.sidebar.file_uploader("Upload your file", type = ["csv"])
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv = pd.read_csv(uploaded_file)
        return csv
    df = load_csv()
    pr = ProfileReport(df, explorative=True)
    st.header("**Input DF**")
    st.write(df)
    st.write("---")
    st.header("**Profiling report with pandas**")
    st_profile_report(pr)
else:
    st.info("Awaiting for CSV file")
    if st.button("press to use example data"):
        def load_data():
            a = pd.DataFrame(np.random.rand(100,5),
                          columns=["age", "banana", "Codenics", "duck", "Ear"])
            return a
        df1 = load_data()
        pr = ProfileReport(df, explorative=True)
        st.Header("**Input DF**")
        st.write(df)
        st.write("---")
        st.header("**Profiling report with pandas**")
        # st_profie_report(pr)    
        
# Insert User Input Parameters
st.sidebar.header("Patient Data")
def user_report():
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
   
    sex_values = ("0", "1")
    cp_values = ("0","1","2","3")
    age = st.sidebar.slider("age", min_value, max_value, value=None, step=None, format=None)
    sex = st.sidebar.selectbox("sex",sex_values)
    cp = st.sidebar.selectbox("Chest pain", cp_values)
    trtbps = st.sidebar.slider("trtbps", min_value_1, max_value_1, value=None, step=None, format=None)
    chol = st.sidebar.slider("chol", min_value_2, max_value_2, value=None, step=None, format=None)
    fbs = st.sidebar.selectbox("fbs", ("1","2","3"))
    restecg = st.sidebar.selectbox("restecg",("0","1"))
    thalachh = st.sidebar.slider("thalachh", min_value_3, max_value_3, value=None, step=None, format=None)
    exng = st.sidebar.selectbox("exng",("0","1"))
    oldpeak = st.sidebar.slider("oldpeak", min_value_4, max_value_4, value=None, step=None, format=None)
    slp = st.sidebar.selectbox("slp",("0","1","2"))
    caa = st.sidebar.selectbox("caa",("0","1","2","3","4"))
    thall = st.sidebar.selectbox("thall",("0","1","2","3"))
    
    
    user_report_data = {"age" : age,
            "sex" : sex,
            "cp" : cp,
            "trtbps" : trtbps,
            "chol" : chol,
            "fbs" : fbs,
            "restecg" : restecg,
            "thalachh" : thalachh,
            "exng" : exng,
            "oldpeak" : oldpeak,
            "slp" : slp,
            "caa" : caa,
            "thall" : thall}
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Description of the Columns
st.write(""" ## Heart attack Analysis dataset's keys definition
         
**1. Age** : Age of the patient

**2. Sex** : Sex of the patient

    Value 0: Female
    
    Value 1: Male

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

# To upload the file
df1 = pd.read_csv("Heart.csv")

# Inserting another subheading
st.subheader("Heart Attack dataset")
st.write(df1)

# Performing EDA
st.subheader("List of Columns")
st.write(df1.columns)
st.subheader("Heart Attack dataset's description")
st.write(df1.describe().T)

# Removing duplicate value from dataset
df1.drop_duplicates(inplace=True)

#gender_option = df1["sex"].unique().tolist()
# sex = st.selectbox("which sex should we plot?", gender_option,0)
# df1 = df1[df1["sex"]==sex]

# Patient data
st.title("Heart Attack prediction")
user_data = user_report()
st.subheader("Patient Data")
st.write

# data splitting into X and y and Train test split
X = df1.drop(["output"], axis=1)
y = df1["output"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
model.fit(X_train, y_train)
user_result = model.predict(user_data)

#output
st.header("Your Report:")
output = ''
if user_result[0]==0:
    output = 'You are safe'
    st.balloons()
else:
    output = "Look after your health"
    st.warning("attack, attack")
st.title(output)
