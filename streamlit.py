import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model (assuming it is a pickle file)
model = pickle.load(open(r"\hyd_house_prediction\House_Model.pkl", 'rb'))  # replace with your actual model file path

# List of places
places = [
    "Upparpally", "Madhapur", "Banjara Hills", "Chandanagar", "Begumpet", "Somajiguda", "Gachibowli", 
    "Manikonda", "Miyapur", "Kondapur", "Hitech City", "Nallagandla", "Financial District", "Anand Nagar Colony",
    "Badangpet", "Mangalpally", "Cherlapally", "Bhanur", "Shaikpet", "Gagillapur", "Mansoorabad", "Mayuri Nagar", 
    "Taramatipet", "Kondamadugu", "Almas Guda", "Uppal", "Bandlaguda", "Appa Junction", "Saidabad", "Kukatpally", 
    "Munganoor", "Old Alwal", "Nagole", "Hastinapuram", "Nizampet", "Nanakramguda", "Kokapet", "Narsingi", 
    "Jubilee Hills"
]

# Initialize label encoder for the places
label_encoder = LabelEncoder()
label_encoder.fit(places)  # Fit the encoder to the list of places

# Set the background image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRya1g7NcmpGf-UFaV21qTGuCEY_l2m6-3mGg&s');
        background-size: cover;
        background-position: center;
    }}
    </style>
    """, unsafe_allow_html=True)

# Set the title with white color
st.markdown("<h1 style='color: white; text-align: center;'>🏠Hyderabad Land Price Predictor</h1>", unsafe_allow_html=True)

# Description section with white color and emojis
st.markdown("""
### 📍 **Welcome to the Land Price Predictor!** 🌍

Select the **location** and input the **area in square feet** to get an estimated land price. 🌱
""", unsafe_allow_html=True)

# Dropdown for selecting place
selected_place = st.selectbox('Select the place:', places)

# Input for second feature (area in square feet)
second_feature_value = st.number_input('Enter the area in square feet:', min_value=1, value=1)

# Function to preprocess input
def preprocess_place(selected_place, second_feature_value):
    encoded_place = label_encoder.transform([selected_place])[0]
    return np.array([encoded_place, second_feature_value]).reshape(1, -1)

# Prediction button and output
if st.button('🔮 **Predict** 🔮'):
    input_data = preprocess_place(selected_place, second_feature_value)
    predicted_price_per_sqft = model.predict(input_data)
    total_price = predicted_price_per_sqft[0] * second_feature_value
    
    # Repeating the predicted price for display
    repeated_price = " + ".join([str(predicted_price_per_sqft[0])] * second_feature_value)
    
    # Display the prediction and price breakdown with white font color
    st.markdown(f"<h3 style='color: white;'>💰 **Predicted Land Price Calculation for {selected_place}:**</h3>", unsafe_allow_html=True)

    # Total price with white color
    st.markdown(f"<p style='color: white; font-size: 20px;'>💵 Total Price = ₹{total_price:.2f}</p>", unsafe_allow_html=True)
