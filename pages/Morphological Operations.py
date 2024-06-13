import streamlit as st
import numpy as np
import cv2

url = "https://github.com/BaranBicen"
if st.sidebar.button("Go to Repository"):
    st.markdown(
        f'<a href="{url}" target="_blank">Go to Repository</a>', unsafe_allow_html=True
    )

kernel1 = np.ones((5, 5), np.uint8)


# Converting to gray scale
def convert_grayscale(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = gray_img / 255.0
    return gray_img


def erosion(img, kernel):
    img = convert_grayscale(img)
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2

    padded_image = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=255
    )

    output_image = np.zeros_like(img)

    for i in range(pad_h, padded_image.shape[0] - pad_h):
        for j in range(pad_w, padded_image.shape[1] - pad_w):
            region = padded_image[i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1]
            output_image[i - pad_h, j - pad_w] = np.min(region[kernel == 1])

    return output_image


def dilation(img, kernel):
    img = convert_grayscale(img)
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2

    padded_image = np.pad(
        img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0
    )

    output_image = np.zeros_like(img)

    for i in range(pad_h, padded_image.shape[0] - pad_h):
        for j in range(pad_w, padded_image.shape[1] - pad_w):
            region = padded_image[i - pad_h : i + pad_h + 1, j - pad_w : j + pad_w + 1]
            output_image[i - pad_h, j - pad_w] = np.max(region[kernel == 1])

    return output_image


def opening(img, kernel):
    eroded_image = erosion(img, kernel)
    opened_image = dilation(eroded_image, kernel)
    return opened_image


def closing(img, kernel):
    dilated_image = dilation(img, kernel)
    closed_image = erosion(dilated_image, kernel)
    return closed_image


"# Morphological Operations"


uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])
if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        "Operations"
    with col2:
        erode = st.button("Erode")
    with col3:
        dilate = st.button("Dilate")
    with col4:
        open = st.button("Open")
    with col5:
        close = st.button("Close")

    if erode:
        "Eroded Image"
        eroded_image = erosion(img, kernel1)
        st.image(eroded_image, use_column_width=True)

    if dilate:
        "Dilated Image"
        dilated_image = dilation(img, kernel1)
        st.image(dilated_image, clamp=True, use_column_width=True)

    if open:
        "Opened Image"
        opened_image = opening(img, kernel1)
        st.image(opened_image, clamp=True, use_column_width=True)

    if close:
        "Closed Image"
        closed_image = closing(img, kernel1)
        st.image(closed_image, clamp=True, use_column_width=True)
