import streamlit as st
from model_helper import predict

st.title("Vechicle Damage")

uploaded_file = st.file_uploader("upload file",type=["jpg","png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path,"wb") as f:
        f.write(uploaded_file.getbuffer())
        st.image(uploaded_file,caption="Uploaded File")
        prediction = predict(image_path)
        st.info(f"Predicted class: {prediction}")