import streamlit as st
import cv2
import numpy as np

url = "https://github.com/BaranBicen"

# Create a button that redirects to the URL
if st.sidebar.button("Go to Repository"):
    st.markdown(
        f'<a href = "{url}" target="_blank">Go to Repository</a>',
        unsafe_allow_html=True,
    )


### Functions
# Converting to grayscale
def convert_grayscale(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = gray_img / 255.0
    return gray_img


###############################################################
# Gaussian blur
def gaussian_blur(img, kernel_size=10, sigma=2.0):
    gray_img = convert_grayscale(img)
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())

    # Apply convolution
    blurred_img = cv2.filter2D(gray_img, -1, kernel)

    return (blurred_img * 255).astype(np.uint8)


"# Blurring"

# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)

    kernel_size = st.slider("Kernel Size", min_value=1, max_value=31, value=10, step=2)
    sigma = st.slider("Sigma", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

    gauss_blur = st.button("Gaussian blur")

    if gauss_blur:
        "Gaussian Blur"
        gauss_blurred = gaussian_blur(img, kernel_size, sigma)
        st.image(gauss_blurred, use_column_width=True)
