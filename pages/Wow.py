import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu
from streamlit_navigation_bar import st_navbar
from scipy.interpolate import UnivariateSpline

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


##########################################################################
# Pencil Sketch Effect
def PencilSketch(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (21, 21), 0, 0)
    final_image = cv2.divide(gray_image, blurred_image, scale=256)
    ret, mask = cv2.threshold(final_image, 70, 255, cv2.THRESH_BINARY)
    sketched_image = cv2.bitwise_and(mask, final_image)
    return sketched_image


###################################################################
# Cartooning Effect
def color_quantization(image, k):
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    success, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result


def cartoon_effect(image):
    quantized_image = color_quantization(image, 7)
    bilateral_image = cv2.bilateralFilter(quantized_image, 8, 150, 150)
    gray_image = cv2.cvtColor(bilateral_image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )
    cartoon_image = cv2.bitwise_and(bilateral_image, bilateral_image, mask=edge_image)
    return cartoon_image


###################################################################
def LUT_8UC1(x, y):
    spl = UnivariateSpline(x, y)
    return spl(range(256))


increase_pixel = LUT_8UC1([0, 64, 128, 192, 256], [0, 70, 140, 210, 256])
decrease_pixel = LUT_8UC1([0, 64, 128, 192, 256], [0, 30, 80, 120, 192])


# Warming and Cooling
def warming_effect(image):
    red, green, blue = cv2.split(image)
    red = cv2.LUT(red, increase_pixel).astype(np.uint8)
    blue = cv2.LUT(blue, decrease_pixel).astype(np.uint8)
    rgb_image = cv2.merge((red, green, blue))
    hue, saturation, value = cv2.split(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV))
    saturation = cv2.LUT(saturation, increase_pixel).astype(np.uint8)
    final_image = cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2RGB)
    return final_image


def cooling_effect(image):
    red, green, blue = cv2.split(image)
    red = cv2.LUT(red, decrease_pixel).astype(np.uint8)
    blue = cv2.LUT(blue, increase_pixel).astype(np.uint8)
    rgb_image = cv2.merge((red, green, blue))
    hue, saturation, value = cv2.split(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV))
    saturation = cv2.LUT(saturation, decrease_pixel).astype(np.uint8)
    final_image = cv2.cvtColor(cv2.merge((hue, saturation, value)), cv2.COLOR_HSV2RGB)
    return final_image


#############################################################################

"# Warming and Cooling Effects"

# To upload images
uploaded_img = st.file_uploader("Select an Image", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    # Convert image to array
    img_arr = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, 1)

    # Display the original image
    st.image(img, channels="BGR", use_column_width=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cartoon = st.button("Cartooning Effect")
    with col2:
        sketch = st.button("Pencil Sketch Effect")
    with col3:
        warm = st.button("Warming Effect")
    with col4:
        cool = st.button("Cooling Effect")

    if cartoon:
        cartoon_img = cartoon_effect(img)
        st.image(cartoon_img, use_column_width=True)
    if sketch:
        sketc_img = PencilSketch(img)
        st.image(sketc_img, use_column_width=True)
    if warm:
        warm_img = warming_effect(img)
        st.image(warm_img, use_column_width=True)
    if cool:
        cool_img = cooling_effect(img)
        st.image(cool_img, use_column_width=True)
