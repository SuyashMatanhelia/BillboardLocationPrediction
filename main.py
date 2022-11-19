import streamlit as st
import geocoder as gc
import requests
from base64 import encodebytes
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


st.title("Billboard Location Predictor (Delhi)")
st.text("This model will predict the best possible locations for your category of bussiness.")
catergory = st.sidebar.selectbox(
    "Type Of Bussiness",
    ("Automobile", "Electronics", "F & B", "Media", "Real Estate")
)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p> 
Project by
</p>
<p> 
<a style='display: ;align: centre; text-align: center;colour: white' href="https://www.linkedin.com/in/arishmit/" target="_blank">Arishmit Ghosh (E20CSE014)</a>
<a style='display: ; text-align: center;' href="https://www.linkedin.com/in/suyashmatanhelia/" target="_blank">Suyash Matanhelia (E20CSE002) </a>
<a style='display: ; text-align: center;' href="https://www.linkedin.com/in/naman-veer-singh-6ab757189/" target="_blank">Naman Veer Singh (E20CSE464)</a>
</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 