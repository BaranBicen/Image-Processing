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


###################################################################################
# Gray image histogram calculation
def calc_gray_hist(img):
    gray_image = convert_grayscale(img)
    histogram = [0] * 256

    for row in gray_image:
        for pixel in row:
            intensity = int(pixel * 255)
            histogram[intensity] = histogram[intensity] + 1
    return histogram


# Colored image histogram calculation
def calc_rgb_hist(img):
    histogram_r = [0] * 256
    histogram_g = [0] * 256
    histogram_b = [0] * 256

    for row in img:
        for pixel in row:
            r, g, b = pixel
            histogram_r[r] += 1
            histogram_g[g] += 1
            histogram_b[b] += 1

    return histogram_r, histogram_g, histogram_b


"# Histogram Applications"

# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])


if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)
    col1, col2 = st.columns(2)
    with col1:
        hist_calc_gray = st.button("Calculate Gray Histogram")
    # Canny button
    with col2:
        hist_calc_rgb = st.button("Calculate RGB Histogram")

    if hist_calc_gray:
        histogram = calc_gray_hist(img)
        fig = go.Figure(data=[go.Bar(x=list(range(256)), y=histogram)])
        fig.update_layout(
            title="Histogram",
            xaxis_title="Intensity",
            yaxis_title="Frequency",
            showlegend=False,
        )
        st.plotly_chart(fig)

    if hist_calc_rgb:
        histogram_r, histogram_g, histogram_b = calc_rgb_hist(img)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=list(range(256)), y=histogram_r, name="Red", marker_color="red")
        )
        fig.add_trace(
            go.Bar(
                x=list(range(256)), y=histogram_g, name="Green", marker_color="green"
            )
        )
        fig.add_trace(
            go.Bar(x=list(range(256)), y=histogram_b, name="Blue", marker_color="blue")
        )

        fig.update_layout(
            title="RGB Histogram",
            xaxis_title="Intensity",
            yaxis_title="Frequency",
            showlegend=True,
        )
        st.plotly_chart(fig)
