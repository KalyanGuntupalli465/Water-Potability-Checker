#Importing the libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
sns.set_style('whitegrid')
from matplotlib import pyplot as plt
import joblib as jb
wpp = jb.load("water_potability_predictor.joblib")
model = wpp['model']
scaler = wpp['scaler']
input_cols = wpp['input_cols']


#setting page configuration
st.set_page_config(page_title='Water Potability Checker', page_icon=':droplet:')


markdown='''
<style>
/* Change button text color to white */
button {
    color: #FFFFFF !important;
}
/* Change text color to white for better visibility */
[data-testid="stAppViewContainer"] {
background-image: linear-gradient(to bottom right, #1976D2, #03A9F4);
    color: #FFFFFF; /* Set text color to white */
}
[data-testid="stHeader"] {
    opacity: 0.0;
    background-image: linear-gradient(to bottom right, #00ACC1, #4DD0E1);
    color: #FFFFFF !important; /* Set text color to white */
}
[data-testid="stSidebar"] {
    background-image: linear-gradient(to bottom right, #00ACC1, #4DD0E1);
}
[class="css-nqowgj edgvbvh3"] {
    border: 1px solid rgb(12, 208, 219);
    border-radius: 5px;
}
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"],
[data-testid="stThumbValue"] {
    color: #FFFFFF !important; /* Set text color to white */
}
[class="main-svg"] {
    opacity: 0.9;
}
</style>

'''

#st.markdown(markdown,unsafe_allow_html=True)


st.write("""# Water Potability Checker""")

st.write("___")



#creating sidebar
st.sidebar.header('Water Quality Metrics')

#Taking user inputs form the side
def user_input_features():
    ph = st.sidebar.slider('ph',0.00, 14.00, 7.04)
    Hardness = st.sidebar.slider('Hardness',47.43,323.12,196.97)
    Solids = st.sidebar.slider('Solids',320.94,61227.20,20927.83)
    Chloramines = st.sidebar.slider('Chloramines',0.35,13.13,7.13 )
    Sulfate = st.sidebar.slider('Sulfate',129.00,481.03,333.39)
    Conductivity = st.sidebar.slider('Conductivity',181.48,753.34,421.88)
    Organic_carbon = st.sidebar.slider('Organic_carbon',2.20,28.30,14.22)
    Trihalomethanes = st.sidebar.slider('Trihalomethanes',0.74,124.00,66.54)
    Turbidity = st.sidebar.slider('Turbidity',1.45,6.74,3.96)
    data = {'ph': ph,
            'Hardness': Hardness,
            'Solids': Solids,
            'Chloramines': Chloramines,
            'Sulfate': Sulfate,
            'Conductivity': Conductivity,
            'Organic_carbon': Organic_carbon,
            'Trihalomethanes': Trihalomethanes,
            'Turbidity': Turbidity}
    features = pd.DataFrame(data, index=[0])
    return features

df1 = user_input_features()

st.write('### Water Quality Metrics :')
st.write(df1)
def predictor(df):
    df[input_cols] = scaler.transform(df[input_cols])
    predictions = model.predict(df[input_cols])
    return predictions[0]

prediction=predictor(df1)

st.write('### Prediction :')
if prediction==0:
    st.write("##### Water is not potabale :warning:")
else:
    st.write("##### Water is potable :innocent:")
st.write("___")
st.write('### Prediction Probability :')
prediction_proba = model.predict_proba(df1)
st.write(prediction_proba)
st.write("0 : Not Potable")
st.write("1 : Potable")


st.write('')
st.write('')
st.write('___')



with st.container():
    left, middle, right = st.columns(3)
       
    with middle:
        st.write(" Developed By:")
        st.write("Mohan Kalyan Guntupalli")
       