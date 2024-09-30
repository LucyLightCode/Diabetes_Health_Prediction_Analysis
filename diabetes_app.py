import pickle
import numpy as np
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(open("project_model.sav", "rb"))

# Function for diabetes prediction
def diabetes_prediction(input_data):
    # Convert the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array for a single instance prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Return a readable result based on the model's prediction
    if prediction[0] == 0:
        return "Non-Diabetes"
    else:
        return "Diabetes"

def main():

    # giving the app a title
    st.title("Diabetes Health Indicators App")
     
    # getting the input data from user
    HighBP =st.text_input("Enter HighBP (0 or 1): ")
    HighChol = st.text_input("Enter HighChol (0 or 1): ")
    BMI = st.text_input("Enter BMI: ")
    Stroke = st.text_input("Enter Stroke (0 or 1): ")
    HeartDiseaseorAttack = st.text_input("Enter HeartDiseaseorAttack (0 or 1): ")
    PhysActivity = st.text_input("Enter PhysActivity (0 or 1): ")
    HvyAlcoholConsump = st.text_input("Enter HvyAlcoholConsump (0 or 1): ")
    AnyHealthcare = st.text_input("Enter AnyHealthcare (0 or 1): ")
    GenHlth = st.text_input("Enter GenHlth (1-5): ")
    DiffWalk = st.text_input("Enter DiffWalk (0 or 1): ")
    Sex = st.text_input("Enter Sex (0 for Female, 1 for Male): ")
    Age = st.text_input("Enter Age (1-13): ")


    # Code for making a prediction
    diabetes = ""

    # Button to trigger the prediction
    if st.button("Diabetes Result"):
        # Call the prediction function with the input data
        diabetes = diabetes_prediction([HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack,
                                        PhysActivity, HvyAlcoholConsump, AnyHealthcare, 
                                        GenHlth, DiffWalk, Sex, Age])
        # Display the prediction result
        st.success(diabetes)

# Run the app
if __name__ == "__main__":
    main()