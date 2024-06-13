import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go

url = "https://github.com/BaranBicen"

# To disable the warnings
st.set_option("deprecation.showPyplotGlobalUse", False)

# Create a button that redirects to the URL
if st.sidebar.button("Go to Repository"):
    st.markdown(
        f'<a href="{url}" target="_blank">Go to Repository</a>', unsafe_allow_html=True
    )


# Converting to gray scale
def convert_grayscale(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = gray_img / 255.0
    return gray_img


# Converting to binary
def convert_binary(img, threshold):
    gray_img = convert_grayscale(img)
    binary_image = (gray_img >= threshold) * 255
    return binary_image


###################################################################


"# Filtering"

# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)
    threshold = st.slider("Threshold", 0, 255, 128)

    col1, col2 = st.columns(2)
    with col1:
        gray = st.button("Apply Grayscale Filter")
    with col2:
        binary = st.button("Apply Binary Filter")

    if gray:
        "Grayscale Image"
        gray_img = convert_grayscale(img)
        st.image(gray_img, use_column_width=True)
    if binary:
        "Binary Image"
        binary_img = convert_binary(img, threshold)
        st.image(binary_img, use_column_width=True)
