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


# Region Growing
def region_growing(image, seed, threshold):
    rows, cols = image.shape
    segmented = np.zeros_like(image)
    segmented[seed[0], seed[1]] = 255
    region_points = []
    region_points.append(seed)

    while len(region_points) > 0:
        x, y = region_points.pop(0)

        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if i >= 0 and j >= 0 and i < rows and j < cols and segmented[i, j] == 0:
                    if abs(int(image[i, j]) - int(image[x, y])) < threshold:
                        segmented[i, j] = 255
                        region_points.append((i, j))

    return segmented


"# Region Growing"

# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 0)

    # Display the original image
    st.image(img, use_column_width=True)

    # Input for seed point
    seed_x = st.number_input(
        "Seed Point X:", min_value=0, max_value=img.shape[0], value=img.shape[0] // 2
    )
    seed_y = st.number_input(
        "Seed Point Y:", min_value=0, max_value=img.shape[1], value=img.shape[1] // 2
    )

    # Input for threshold
    threshold = st.number_input("Threshold:", min_value=1, max_value=255, value=10)

    if st.button("Perform Region Growing"):
        seed = (int(seed_x), int(seed_y))
        segmented_img = region_growing(img, seed, threshold)
        st.image(segmented_img, use_column_width=True)
