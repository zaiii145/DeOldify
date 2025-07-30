import numpy as np
import cv2
import streamlit as st
import os
from io import BytesIO
from PIL import Image

PROTOTXT = "COLOURIZATION/colorization_deploy_v2.prototxt"
POINTS = "COLOURIZATION/pts_in_hull.npy"
MODEL = "COLOURIZATION/colorization_release_v2.caffemodel"

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def colorize_image(image):
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")
    
    return colorized


st.title("DeOldify")

uploaded_file = st.file_uploader("Upload a black-and-white image", type=["jpg", "jpeg", "png","webp"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    colorized_cv = colorize_image(image_cv)
    colorized = Image.fromarray(cv2.cvtColor(colorized_cv, cv2.COLOR_BGR2RGB))

    col1, col2 = st.columns(2)
    col1.image(image, caption="Original Image", use_container_width=True)
    col2.image(colorized, caption="Colorised Image", use_container_width=True)

    buf = BytesIO()
    colorized.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Colorised Image",
        data=byte_im,
        file_name="colorised.png",
        mime="image/png"
    )
