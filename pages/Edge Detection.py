import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


url = "https://github.com/BaranBicen"

# Create a button that redirects to the URL
if st.sidebar.button("Go to Repository"):
    st.markdown(
        f'<a href="{url}" target="_blank">Go to Repository</a>', unsafe_allow_html=True
    )

### Filters


# Converting to gray scale
def convert_grayscale(img):
    gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    gray_img = gray_img / 255.0
    return gray_img


######################################################################################################
# Functions for sobel
# Convulution for sobel edge detection
def convolve2d_s(image, kernel):
    kernel = np.flipud(np.fliplr(kernel))
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x] = (kernel * image_padded[y : y + 3, x : x + 3]).sum()
    return output


# Sobel kernels
def sobel_operator(image):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolve2d_s(image, Gx)
    gradient_y = convolve2d_s(image, Gy)

    return gradient_x, gradient_y


# Gradient
def gradient_magnitude(gradient_x, gradient_y):
    return np.sqrt(gradient_x**2 + gradient_y**2)


# Treshold
def apply_threshold(gradient_magnitude, threshold):
    scaled_grad_mag = (gradient_magnitude / np.max(gradient_magnitude)) * 255
    edges = (scaled_grad_mag > threshold) * 255
    return edges.astype(np.uint8)


# Main function
def sobel_edge_detection(image, threshold=50):
    gray_image = convert_grayscale(image)
    gradient_x, gradient_y = sobel_operator(gray_image)
    grad_mag = gradient_magnitude(gradient_x, gradient_y)
    edges = apply_threshold(grad_mag, threshold)
    return edges


#######################################################################################################


#######################################################################################################
# Canny
# Gaussian blur for noise reduction
def gaussian_blur(img, kernel_size=5, sigma=1.0):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2))
        * np.exp(
            -((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)
            / (2 * sigma**2)
        ),
        (kernel_size, kernel_size),
    )

    kernel = kernel / np.sum(kernel)

    blurred_img = cv2.filter2D(img, -1, kernel)

    return blurred_img


# Canny edge detection
def canny_edge_detection(img):
    gray_img = convert_grayscale(img)
    blurred_img = gaussian_blur(gray_img)
    gradient_x, gradient_y = sobel_operator(gray_img)
    theta = np.arctan(gradient_y, gradient_x)
    theta_normalized = (theta - np.min(theta)) / (np.max(theta) - np.min(theta))
    return theta_normalized


# Main Part
# Page title
"# Edge Detection"
# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)

    ### Fiter Buttons
    col1, col2 = st.columns(2)
    # Sobel Button
    with col1:
        sobel = st.button("Sobel")
    # Canny button
    with col2:
        canny = st.button("Canny")

    # If sobel sutton is clicked
    if sobel:
        "Sobel"
        edges = sobel_edge_detection(img)
        st.image(edges, use_column_width=True)
    # If canny button is clicked
    if canny:
        "Canny"
        img_edge = canny_edge_detection(img)
        st.image(img_edge, use_column_width=True)
