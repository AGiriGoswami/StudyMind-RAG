import streamlit as st


api_key = st.secrets["GOOGLE_API_KEY"]

st.title("📚 StudyMind")

uploaded_file = st.file_uploader("Upload PDF")

question = st.text_input("Ask Question")

if st.button("Generate Answer"):
    st.write("Answer Generated")