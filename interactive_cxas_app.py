import streamlit as st
import subprocess
import os
from PIL import Image
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_DIR, "tmp")
OUTPUT_DIR = os.path.join(TMP_DIR, "output")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_segmentation(input_image_path, output_folder, mode="segment", gpu="cpu"):
    # Paksa env variable agar library tidak bingung mencari resource
    command = f"cxas -i {input_image_path} -o {output_folder} --mode {mode} -g {gpu} -s"
    result = subprocess.run(
        command, shell=True, env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        st.error(f"Error Command: {result.stderr}")
    return result


def colorize_and_outline_mask(mask_image, color=(0, 255, 0)):
    mask_np = np.array(mask_image.convert("L"))
    _, mask_np = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    colorized_mask = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    colorized_mask[mask_np == 255] = color
    edges = cv2.Canny(mask_np, 100, 200)
    colorized_mask[edges == 255] = [255, 255, 255]
    return colorized_mask


def overlay_mask_on_image(input_image, mask_image, alpha=0.5):
    input_image_np = np.array(input_image)
    if len(input_image_np.shape) == 2:
        input_image_np = cv2.cvtColor(input_image_np, cv2.COLOR_GRAY2RGB)
    mask_image_resized = cv2.resize(
        mask_image, (input_image_np.shape[1], input_image_np.shape[0])
    )
    overlayed_image = cv2.addWeighted(
        input_image_np, 1 - alpha, mask_image_resized, alpha, 0
    )
    return overlayed_image


st.set_page_config(page_title="Chest X-Ray Segmentation", layout="wide")
st.title("🫁 Image Segmentation Tool (CXAS)")

if "segmentation_done" not in st.session_state:
    st.session_state.segmentation_done = False
    st.session_state.mask_files = []
    st.session_state.output_folder = ""

uploaded_image = st.file_uploader(
    "Upload Chest X-Ray (JPG/PNG)", type=["png", "jpg", "jpeg"]
)

if uploaded_image is not None:
    input_image_name = os.path.splitext(uploaded_image.name)[0]
    input_image_path = os.path.join(TMP_DIR, uploaded_image.name)

    img = Image.open(uploaded_image)
    img.save(input_image_path)

    specific_output_path = os.path.join(OUTPUT_DIR, input_image_name)

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.image(img, caption="Original Image", width="stretch")

        if not st.session_state.segmentation_done:
            if st.button("🚀 Run Segmentation"):
                with st.spinner("Processing... Please wait."):
                    print("Lagi run segmentasi")
                    run_segmentation(input_image_path, OUTPUT_DIR)

                    print("Otw nyimpen output")
                    if os.path.exists(specific_output_path):
                        st.session_state.output_folder = specific_output_path
                        st.session_state.mask_files = [
                            f
                            for f in os.listdir(specific_output_path)
                            if f.endswith(".png")
                        ]
                        st.session_state.segmentation_done = True
                        st.rerun()
                    else:
                        st.error(
                            f"Folder hasil tidak ditemukan di: {specific_output_path}. Cek log terminal untuk detail error library 'cxas'."
                        )

    if st.session_state.segmentation_done:
        with col_result:
            if st.session_state.mask_files:
                selected_mask = st.selectbox(
                    "Pilih Masker Anatomi:", st.session_state.mask_files
                )
                mask_path = os.path.join(st.session_state.output_folder, selected_mask)
                mask_img = Image.open(mask_path)
                color_mask = colorize_and_outline_mask(mask_img)
                result_img = overlay_mask_on_image(img, color_mask)

                st.image(
                    result_img, caption=f"Overlay: {selected_mask}", width="stretch"
                )
                st.success("Segmentasi Berhasil!")
