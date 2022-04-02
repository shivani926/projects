import streamlit as st
import pickle
st.title("HOUSE PRICE PREDICTION")
st.title("shivani tiwari")
pickle_in=open('model','rb')
xgb=pickle.load(pickle_in)
num1=st.number_input("enter Carpate Area",key="1")
num2=st.number_input("Enter Full bath",key="2")

if st.button("predict"):
    pred=str(xgb.predict([[num1,num2]]))
    st.success("price predict: "+pred)