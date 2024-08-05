from langchain_groq import ChatGroq
import streamlit as st



chat = ChatGroq(temperature=0, groq_api_key=st.secrets["groq_api_key"], model_name="mixtral-8x7b-32768")
print(chat('hi'))