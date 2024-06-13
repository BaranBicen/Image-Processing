import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
import pandas as pd

st.set_page_config(
    page_title="OpenCV Streamlit App",
    page_icon="👋",
)

url = "https://github.com/BaranBicen"

# Create a button that redirects to the URL
if st.sidebar.button("Go to Repository"):
    st.markdown(
        f'<a href="{url}" target="_blank">Go to Repository</a>', unsafe_allow_html=True
    )

###################################################################################


# Histogram Equalization
def histogram_equalization(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    cdf = histogram.cumsum()

    cdf_normalized = cdf * float(histogram.max()) / cdf.max()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")

    equalized_image = cdf[image]

    return equalized_image


########################################################################################


# Converting to gray scale
def convert_grayscale(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = gray_img / 255.0
    return gray_img


def calc_gray_hist(img):
    gray_image = convert_grayscale(img)
    histogram = [0] * 256

    for row in gray_image:
        for pixel in row:
            intensity = int(pixel * 255)
            histogram[intensity] = histogram[intensity] + 1
    return histogram


def hist_strech(img):
    img = img.astype(np.float32)

    I_min = np.min(img)
    I_max = np.max(img)

    I_min_new = 0
    I_max_new = 255

    stretched_img = ((img - I_min) / (I_max - I_min)) * (
        I_max_new - I_min_new
    ) + I_min_new
    stretched_img = np.clip(stretched_img, 0, 255).astype(np.uint8)

    return stretched_img


# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    st.image(img, channels="BGR", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        strech = st.button("Streching")
    with col2:
        equal = st.button("Equalize Histogram")

    if strech:
        hist_streched = hist_strech(img)
        st.image(hist_streched, use_column_width=True)
    if equal:
        hist_equilized = histogram_equalization(img)
        st.image(hist_equilized, use_column_width=True)
