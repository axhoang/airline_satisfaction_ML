import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Title of the Streamlit app
st.title('Customer Satisfaction Prediction')

# Subtitle
st.subheader('Predicting if a customer is likely to be satisfied based on flight experience')

# Input features
st.markdown("### Input Features")
st.markdown("Please adjust the sliders and input fields to match the flight details.")

inflight_entertainment = st.slider('Inflight Entertainment', 0, 5, 3, help="Rating of inflight entertainment (0-5)")
on_board_service = st.slider('On-board Service', 0, 5, 3, help="Rating of on-board service (0-5)")
seat_comfort = st.slider('Seat Comfort', 0, 5, 3, help="Rating of seat comfort (0-5)")
cleanliness = st.slider('Cleanliness', 0, 5, 3, help="Rating of cleanliness (0-5)")
departure_delay = st.number_input('Departure Delay in Minutes', 0, 500, 0, help="Minutes of departure delay")

# Add a button to make the prediction
if st.button('Predict'):
    # Make prediction
    input_data = np.array([[inflight_entertainment, on_board_service, seat_comfort, cleanliness, departure_delay]])
    prediction = model.predict(input_data)[0]
    
    # Display result
    if prediction == 1:
        st.success('The customer is likely to be satisfied.')
    else:
        st.warning('The customer is likely to be neutral or dissatisfied.')
