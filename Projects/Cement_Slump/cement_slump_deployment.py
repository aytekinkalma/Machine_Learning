# !pip install streamlit

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.title("Machine learning Prediction Project in Streamlit")

# !pip install sklearn
import sklearn
import pickle
filename = 'final_model_hearing'
model = pickle.load(open(filename, 'rb'))

Cement = st.sidebar.number_input("Cement:",min_value=137, max_value=374)
Slag = st.sidebar.number_input("Slag:",min_value=0, max_value=193)
Fly_ash = st.sidebar.number_input("Fly ash:",min_value=0, max_value=260)
Water = st.sidebar.number_input("Water:",min_value=160, max_value=240)
SP = st.sidebar.number_input("SP:",min_value=5, max_value=19)
Coarse_Aggr = st.sidebar.number_input("Coarse Aggr.:",min_value=708, max_value=1049)
Fine_Aggr = st.sidebar.number_input("Fine Aggr.:",min_value=640, max_value=902)
SLUMP = st.sidebar.number_input("SLUMP(cm):",min_value=0, max_value=29)
FLOW = st.sidebar.number_input("FLOW(cm):",min_value=50, max_value=78)






my_dict = {"Cement":Cement,
           "Slag":Slag,
           "Fly ash":Fly_ash,
           "Water":Water,
           "SP":SP,
           "Coarse Aggr.":Coarse_Aggr,
           "Fine Aggr.": Fine_Aggr,
           "SLUMP(cm)":SLUMP,
           "FLOW(cm)":FLOW
           }

df = pd.DataFrame.from_dict([my_dict])

#cols = {"hp_kW": "Horse Power", "Age": "Age", "km": "km Traveled","Gears": "Gears"}

df_show = df.copy()
#df_show.rename(columns = cols, inplace = True)
st.write("Selected Specs: \n")
st.table(df_show)

if st.button("Predict"):
    pred = model.predict(df)
    pred


st.write("\n\n")


