import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title = "Ice Cream Sales Predictor",page_icon="🍦",layout= "centered")
 
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(to top, #022f40, #38aecc);
    }
    .main {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
    """,unsafe_allow_html=True)

st.title("🍦Ice Cream Sales Predictor")

st.markdown("### Predict ice cream sales based on temperature 🧊")

a = pd.read_csv("iceVtemp.csv")
df = pd.DataFrame(a)
X,y = df[["Temperature"]],df[["Ice Cream Profits"]]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)

temp_input = st.slider("Select Temperature (°F)", min_value=40, max_value=110, value=65)

input_df = pd.DataFrame({"Temperature": [temp_input]})
y_pred = model.predict(input_df)[0][0]
st.metric(label="Predicted Ice Cream Sales 💰", value=f"₹{y_pred:.2f}")

fig,pt = plt.subplots()
pt.scatter(X,y,color="blue",label="Actual Data")
pt.plot(X,model.predict(X),color="red",label = "Regression Line")
pt.set_xlabel("Temperature (°F)")
pt.set_ylabel("Sales (₹)")
pt.set_title("Temperature VS Ice Cream Sales",fontsize=14)
pt.legend()

st.pyplot(fig)
