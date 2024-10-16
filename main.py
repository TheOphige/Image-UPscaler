import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionUpscalePipeline
import io

# Load the model
@st.cache_resource
def load_model():
    model_id = "nateraw/real-esrgan"
    model = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model

# Function to upscale image
def upscale_image(model, image, scale_factor):
    # Convert image to correct format for model
    image = image.convert("RGB")
    image = image.resize((image.width * scale_factor, image.height * scale_factor), Image.BICUBIC)
    
    # Pass image to model
    upscaled_image = model(image, guidance_scale=7.5).images[0]
    
    return upscaled_image

# Streamlit app interface
st.title("Image Upscaling with Real-ESRGAN")
st.write("Upload an image and use the slider to adjust the scaling factor (2x, 4x).")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Scaling factor slider
    scale_factor = st.slider("Select scale factor", 1, 4, 2)  # Default is 2x

    # Load Real-ESRGAN model
    model = load_model()

    # Automatically upscale the image as the slider moves
    with st.spinner("Upscaling..."):
        upscaled_image = upscale_image(model, image, scale_factor)
    
    # Display the upscaled image
    st.image(upscaled_image, caption=f"Upscaled Image ({scale_factor}x)", use_column_width=True)

    # Option to download the upscaled image
    buf = io.BytesIO()
    upscaled_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(label="Download Upscaled Image", data=byte_im, file_name="upscaled_image.png", mime="image/png")
