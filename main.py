import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("First one 98/Train model/trained_plant_disease_model_ahmed.keras")
    # model = tf.keras.models.load_model("Secound One/Trained/trained_plant_Lower_model_ahmed.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)  # return index of max element
    predicted_probabilities = predictions[0]  # get the probabilities for the current image
    return predicted_index, predicted_probabilities  # return both the index and probabilities

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets while preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                #### Team Members
                1. Ahmed Hamdy Saeed 
                2. Yousef Ahmed
                3. George Bebawy
                """)

# Disease descriptions
disease_descriptions = {
    'Apple___Apple_scab': "Apple scab is a fungal disease that affects the leaves and fruit of apple trees, causing dark, olive-green lesions.",
    'Apple___Black_rot': "Black rot is caused by a fungus that leads to dark, sunken lesions on fruit and can cause premature fruit drop.",
    'Apple___Cedar_apple_rust': "Cedar apple rust is a disease that affects apples and is characterized by yellow spots on leaves and fruit.",
    'Apple___healthy': "The plant is healthy and shows no signs of disease.",
    'Blueberry___healthy': "The blueberry plant is healthy and free from disease.",
    'Cherry_(including_sour)___Powdery_mildew': "Powdery mildew is a fungal disease that appears as a white powdery coating on leaves and stems.",
    'Cherry_(including_sour)___healthy': "The cherry plant is healthy and exhibits no disease symptoms.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Cercospora leaf spot causes grayish spots on the leaves, reducing photosynthesis.",
    'Corn_(maize)___Common_rust_': "Common rust is a fungal disease characterized by orange or reddish-brown pustules on leaves.",
    'Corn_(maize)___Northern_Leaf_Blight': "Northern leaf blight is a fungal disease that causes long, grayish-brown lesions on leaves.",
    'Corn_(maize)___healthy': "The corn plant is healthy and shows no signs of disease.",
    'Grape___Black_rot': "Black rot is a disease that causes dark lesions on grape leaves and can affect fruit quality.",
    'Grape___Esca_(Black_Measles)': "Esca is a serious grapevine disease causing leaf necrosis and decline of vines.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Leaf blight is a fungal disease leading to leaf spots and reduced yield.",
    'Grape___healthy': "The grape plant is healthy and unaffected by diseases.",
    'Orange___Haunglongbing_(Citrus_greening)': "Citrus greening is a bacterial disease that causes yellowing of leaves and poor fruit quality.",
    'Peach___Bacterial_spot': "Bacterial spot causes dark lesions on leaves and fruit, affecting peach quality.",
    'Peach___healthy': "The peach plant is healthy and exhibits no disease symptoms.",
    'Pepper,_bell___Bacterial_spot': "Bacterial spot leads to dark, water-soaked lesions on leaves and fruits of pepper plants.",
    'Pepper,_bell___healthy': "The bell pepper plant is healthy and free from disease.",
    'Potato___Early_blight': "Early blight is a fungal disease causing dark, concentric rings on leaves.",
    'Potato___Late_blight': "Late blight is a serious disease that can destroy potato crops, causing dark spots and decay.",
    'Potato___healthy': "The potato plant is healthy and shows no signs of disease.",
    'Raspberry___healthy': "The raspberry plant is healthy and unaffected by diseases.",
    'Soybean___healthy': "The soybean plant is healthy and free from disease.",
    'Squash___Powdery_mildew': "Powdery mildew appears as a white coating on squash leaves, affecting growth.",
    'Strawberry___Leaf_scorch': "Leaf scorch leads to brown, crispy leaf edges and can weaken the plant.",
    'Strawberry___healthy': "The strawberry plant is healthy and exhibits no disease symptoms.",
    'Tomato___Bacterial_spot': "Bacterial spot causes dark lesions on tomato leaves and fruits.",
    'Tomato___Early_blight': "Early blight is characterized by dark spots and concentric rings on leaves.",
    'Tomato___Late_blight': "Late blight causes dark, water-soaked spots on leaves, often leading to plant death.",
    'Tomato___Leaf_Mold': "Leaf mold appears as a fuzzy, grayish growth on the upper side of tomato leaves.",
    'Tomato___Septoria_leaf_spot': "Septoria leaf spot leads to small, dark spots on leaves, affecting growth.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Spider mites cause stippling on leaves and can lead to leaf drop.",
    'Tomato___Target_Spot': "Target spot is characterized by dark, concentric spots on leaves, reducing photosynthesis.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "This viral disease causes yellowing and curling of tomato leaves.",
    'Tomato___Tomato_mosaic_virus': "Mosaic virus causes mottled leaves and stunted growth in tomato plants.",
    'Tomato___healthy': "The tomato plant is healthy and free from disease."
}

# Prediction Page
if app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    # Create a two-column layout
    col1, col2 = st.columns([3, 1])  # Adjust the ratio as needed
    
    with col1:
        if st.button("Show Image") and test_image is not None:
            st.image(test_image, width=4, use_column_width=True)
    
    with col2:
        if st.button("Predict") and test_image is not None:
            st.write("Our Prediction")
            
            result_index, predicted_probabilities = model_prediction(test_image)
            
            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']
            
            predicted_class = class_name[result_index]
            plant_type, disease_type = predicted_class.split('___')
            predicted_percentage = np.max(predicted_probabilities) * 100  # Get the maximum probability
            
            # Display predictions on the same line, right-aligned
            st.success(f"Predicted Plant Type: {plant_type}")
            st.success(f"Type of Disease: {disease_type}")
            st.success(f"Prediction Confidence: {predicted_percentage:.2f}%")  # Updated style
            
            # Disease details section with expander
            if predicted_class in disease_descriptions:
                with st.expander("Show Disease Details"):
                    st.write(disease_descriptions[predicted_class])  # Description appears when expanded

