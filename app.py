from streamlit_option_menu import option_menu
import streamlit as st
import subprocess
import pandas as pd
import os
import numpy as np




st.markdown('### Wafer Fault  prediction AI Modal')
with st.sidebar:
    selected = option_menu("Choose an Option", ["New Training", 'Predict'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    
if selected=="New Training":

    
    st.write("Modal Training Wizard") 
    if st.button("Run Training"):
        output_placeholder = st.empty()
        output_placeholder.text("Training in progress...")

        try:
            # Run the training script using subprocess
            result = subprocess.run(["python", "training.py"], capture_output=True, text=True, check=True)

            # Display the output of the training script dynamically
            output_placeholder.text("Training Script Output:")
            output_placeholder.text(result.stdout)

            # Display a message indicating that training is complete
            st.success("Training complete!")
        except subprocess.CalledProcessError as e:
            # Display an error message if the training script fails
            st.error(f"Training failed with error code {e.returncode}")
            output_placeholder.text("Training Script Errors:")
            output_placeholder.text(e.stderr)

if selected=="Predict":
    

    st.title("DataFrame Upload Sample")

    # Initialize df outside the block
    df = None

    # File uploader widget
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Process the uploaded file
        try:
            if uploaded_file.type == "application/vnd.ms-excel":
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)

            # Display the DataFrame
            st.dataframe(df, height=300)

            # Additional processing or analysis can be done with the DataFrame
            # For example, you can run your prediction model on this data

        except Exception as e:
            st.error(f"Error: {e}")

    # You can use df outside the if-else block
    if df is not None:
        # Perform additional operations with the DataFrame
        st.markdown("<h2>DataFrame is available for further processing.</h2>", unsafe_allow_html=True)


        

# Assuming you have a DataFrame named df

# Create the folder if it doesn't exist
        folder_path = 'Temp_data'
        os.makedirs(folder_path, exist_ok=True)
        df.to_csv(os.path.join(folder_path, 'your_data.csv'), index=False)


        if st.button("Run Prediction"):
            output_placeholder = st.empty()
            output_placeholder.text("Prediction in progress...")

            try:
                # Run the training script using subprocess
                result = subprocess.run(["python", "prediction.py"], capture_output=True, text=True, check=True)

                # Display the output of the training script dynamically
                output_placeholder.text("Training Script Output:")
                output_placeholder.text(result.stdout)

                # Display a message indicating that training is complete
                st.success("Prediction complete!")
            except subprocess.CalledProcessError as e:
                # Display an error message if the training script fails
                st.error(f"Training failed with error code {e.returncode}")
                output_placeholder.text("Training Script Errors:")
                output_placeholder.text(e.stderr)


        
