from operator import index
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import subprocess
from ImageProcessing import image_processing
from Model_Building import model_generation


width =100
height=100
# Function to perform ancient script recognition
def recognize_ancient_script(image):
    # Placeholder function, replace with your actual model code
    # This function should take an image as input and return the results
    # You can replace it with your trained model for ancient script recognition
    # For the sake of example, let's assume it returns the same image for now

    image.save("original.jpg")
    
    # Execute the notebook imageprocessing.ipynb
    #subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "Tamil-Epigraphs-data-creation-and-recognition-main\image preprocessing (cutting the characters and creating dataset) .ipynb"])
    
    image_processing()
    print('Processing Image Completed')
    model_generation()
        
    print("Displaying Input and Output ..............")
    st.subheader("Processed Image")
    

# Load an image
    processed_image = Image.open(r'C:/Users/praveena/Downloads/Tamil-Epigraphs-data-creation-and-recognition-main (1)/Tamil-Epigraphs-data-creation-and-recognition-main/input.jpg')
    st.image(processed_image,caption='Processed Image')

    
    output_image=Image.open(r"C:\Users\praveena\Downloads\Tamil-Epigraphs-data-creation-and-recognition-main (1)\Tamil-Epigraphs-data-creation-and-recognition-main\output.jpg")


    return output_image

def main():
    st.title('Ancient Script Recognition')

    st.write("Upload an image containing ancient script")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        if uploaded_file.type not in ["image/jpeg", "image/png","image/jpg"]:
            st.error("Only JPG, JPEG, and PNG file formats are supported.")
        else:    
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image')

           # Perform ancient script recognition
            result_image = recognize_ancient_script(image)

        # Display the result
            st.subheader('Result')
            st.image(result_image, caption='Recognized Script')

            print("Process Completed")

if __name__ == "__main__":
    main()
