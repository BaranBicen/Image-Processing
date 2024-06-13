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

"# Hawdy Mate!"


"""Welcome to our Streamlit app! Today, we're diving into the fascinating world of image processing without relying on OpenCV's built-in functions. It's a bit like exploring a new frontier in the digital realm.

Imagine you're handed a picture, just a grid of colored pixels. With OpenCV, you might use its powerful tools to manipulate, enhance, or extract information from that image. But what if we took a different approach?

Instead of relying on pre-built functions, we're rolling up our sleeves and delving into the raw data ourselves. It's like being an artist with a blank canvas, crafting our own tools to paint our digital masterpiece.

Sure, it's a bit more challenging. We're not taking the well-trodden path here. But isn't that where innovation thrives? By breaking away from the constraints of pre-packaged solutions, we're free to explore new techniques, experiment with novel approaches, and truly understand the inner workings of image processing.

So, as you navigate through our app, keep in mind the spirit of exploration and discovery that drives us. Every pixel, every line of code, is a step forward in our journey to unlock the full potential of image processing, on our own terms.

Let's embark on this adventure together and see where it takes us. Welcome to the world of image processing, reimagined.

"""

"# How To Use"

"""To use this app simply choose the operations you'd like to make from the side bar and upload an image from you computer. Upon uploading your image you will see a set of buttons where you can click and see the filter's, histogram equalization's
etc. results. The most important thing is to having fun and learning."""

"# Contribution"
"""If you want to contribute or simply make a better version you most ceartinly can. The source code is provided in my github @BaranBicen or click the github from the sidebar."""

# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)

    st.write("Matrix Representation")
    st.write(img)
