import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (160, 160)
MODEL_PATH = "sea_animals_cnn.keras"
CLASS_PATH = "class_names.json"

st.set_page_config(page_title="Klasifikasi Hewan Laut", page_icon="🐠")


@st.cache_resource
def load_resources():
    # compile=False: untuk inferensi tidak perlu fungsi loss kustom.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(img: Image.Image) -> np.ndarray:
    # Sama seperti saat training: RGB, resize bilinear, nilai 0-255 (float32).
    arr = tf.convert_to_tensor(np.array(img.convert("RGB")))
    arr = tf.image.resize(arr, IMG_SIZE)
    return tf.cast(arr, tf.float32).numpy()


def zoom_in(batch, faktor=0.90):
    hasil = tf.image.central_crop(batch, central_fraction=faktor)
    return tf.image.resize(hasil, IMG_SIZE)


def predict(model, arr, use_tta=True):
    batch = tf.convert_to_tensor(arr[None, ...])
    if not use_tta:
        return model.predict(batch, verbose=0)[0]
    zoom = zoom_in(batch)
    variasi = [batch, tf.image.flip_left_right(batch),
               zoom, tf.image.flip_left_right(zoom)]
    return np.mean([model.predict(v, verbose=0) for v in variasi], axis=0)[0]


def nama_tampil(nama):
    return nama.replace("_", " / ").title()


st.title("🐠 Klasifikasi 10 Hewan Laut")
st.write("Custom CNN (residual) — unggah foto hewan laut untuk diprediksi.")

model, class_names = load_resources()

with st.sidebar:
    st.header("Pengaturan")
    use_tta = st.checkbox("Gunakan Test-Time Augmentation (TTA)", value=True)
    st.caption("Kelas: " + ", ".join(nama_tampil(c) for c in class_names))

file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png", "webp", "bmp"])

if file:
    img = Image.open(file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    with st.spinner("Memprediksi..."):
        probs = predict(model, preprocess(img), use_tta)

    top = int(np.argmax(probs))
    st.success(f"Prediksi: **{nama_tampil(class_names[top])}** ({probs[top]*100:.1f}%)")

    urut = np.argsort(probs)[::-1]
    st.subheader("Probabilitas tiap kelas")
    st.bar_chart(
        {nama_tampil(class_names[i]): float(probs[i] * 100) for i in urut},
        horizontal=True,
    )
