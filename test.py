import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import emoji
import keras

# Define the URL of the model file on Google Drive and the output path
model_url = 'https://drive.google.com/uc?id=12077yIkpCaAkE8gb-H_Kq7SCXtv6sDPs'
output_path = 'my_model2.hdf5'

# Download the model from Google Drive
gdown.download(model_url, output_path, quiet=False)
model = keras.models.load_model(output_path)

# Define the Streamlit app
def app():
    st.title('NutriScore: A Deep Learning-based Food Classification System')
    
    # Load the pre-trained model
    
    
    # Define the user input
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    score = 0
    
    for uploaded_file in uploaded_files:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Resize the image to the input size of the model
        image = image.resize((224, 224))
        
        # Convert the image to a numpy array
        image_array = np.array(image)
        
        # Preprocess the image
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make a prediction using the deep learning model
        prediction = model.predict(image_array)
        
        class_names = ['Healthy', 'Unhealthy']
        prediction_class = class_names[np.argmax(prediction)]
        
        if prediction_class == "Unhealthy":
            score -= 1
            st.write('This image is Unhealthy.', emoji.emojize(":disappointed_face:"))
        elif prediction_class == "Healthy":
            score += 1
            st.write("Hello! :wave: This image is healthy. :smile:")
        else:
            print("Unexpected prediction class:", prediction_class)

    if score > 0:
        st.write('Yeah!! Final Score:', score)
        ss = emoji.emojize(":star-struck:")
        st.write(f'<span style="font-size: 3rem">{ss}</span>', unsafe_allow_html=True)
    elif score < 0:
        st.write('Final Score:', score)
        sob1 = emoji.emojize(":sob:")
        st.write(f'<span style="font-size: 3rem">{sob1}</span>', unsafe_allow_html=True)
    elif score == 0:
        st.write('Final Score:', score)
        nef = emoji.emojize(":neutral_face:")
        st.write(f'<span style="font-size: 3rem">{nef}</span>', unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    app()
