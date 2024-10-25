from typing_extensions import Text
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import cvlib as cv
import cv2
import sys
import numpy as np
from cvlib.object_detection import draw_bbox
from tempfile import NamedTemporaryFile


# # Set page configuration and hide headers/footers
st.set_page_config(page_title="DetectiNator", layout="wide", initial_sidebar_state="collapsed")
hide_decoration_bar_style = '''<style>header {visibility: hidden;}
</style><style>footer{visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# <style> .main {overflow: hidden} </style>
# tab1,tab2 = st.tabs(["static","real-time"])
# with tab1:
st.title("Image detection and captioning")

with st.container(border =True,height = 705):
    cola,colb = st.columns(2)
    with cola:
        with st.container(border = True,height = 670):
            uploaded_img = st.file_uploader("choose an image", type=['png','jpg'])
            if uploaded_img is not None:
                with NamedTemporaryFile(dir='.', suffix='.csv') as f:
                    f.write(uploaded_img.getbuffer())
                    img = cv2.imread(f.name,)
            else:
                st.write("Please upload an image")

            # selected_model = st.radio("Select a model", ["yolov4"])
            selected_model = "yolov4"
            if st.button("Detect"):
                desired_width = 800
                aspect_ratio = desired_width / img.shape[1]
                desired_height = int(img.shape[0] * aspect_ratio)
                dim = (desired_width, desired_height)
                img_resize = cv2.resize(img,dsize=dim,interpolation=cv2.INTER_AREA)

                bbox, label ,conf = cv.detect_common_objects(img_resize,model = selected_model)
                # print(bbox, label, conf)
                label_name = label[0]
                confidence = conf[0]*100
                st.write(f"Detected {label_name}, Confidence = { confidence}")
                out_img = draw_bbox(img_resize,bbox,label,conf)
                out_img_RGB = cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB)
                with colb:
                    with st.container(border = True,height = 670):
                        st.header("Objects detected using YOLOV4")
                        st.image(out_img_RGB,width = 450)
                        annotated_img = Image.fromarray(out_img_RGB)
                        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                        prompt = "this is an annotated image which has detected some objects using an object detection algorithm, define the annotated objects with some information about the surrounding"
                        inputs = processor(images=annotated_img,return_tensors="pt")
                        out = model.generate(**inputs,max_new_tokens = 50)
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        st.header(caption)
                        st.caption("caption generated using BLIP[Huggingface transformer lib]")

                # with col3:
                #     # Chat with the model based on the caption
                #         st.subheader("Chat with the Model")
                #         user_input = st.text_input("Ask about the image or caption:")
                #         if user_input:
                #             # Simple response generation logic (for demonstration)
                #             # In a real implementation, this would involve model inference
                #             response = f"You asked: '{user_input}' about the caption: '{caption}'."
                #             st.write(response)
    # cv2.imwrite("object_detection.jpg", out_img)
    # cv2.namedWindow("Object_detected")
    # cv2.moveWindow("Object_detected",700,200)
    # cv2.imshow("Object_detected", out_img)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # img_file_buffer = st.camera_input("Take a picture")

    # if img_file_buffer is not None:
    #     # To read image file buffer with OpenCV:
    #     bytes_data = img_file_buffer.getvalue()
    #     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    #     # Check the type of cv2_img:
    #     # Should output: <class 'numpy.ndarray'>
    #     st.write(type(cv2_img))

    #     # Check the shape of cv2_img:
    #     # Should output shape: (height, width, channels)
    #     st.write(cv2_img.shape)
