# filename: image_enhance_app.py

import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import tempfile

def enhance_image(pil_img, bright=1.2, contrast=1.3, sharp=2.0, denoise_h=10):
    # Enhance with PIL
    img = pil_img.copy()
    img = ImageEnhance.Brightness(img).enhance(bright)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    img = ImageEnhance.Sharpness(img).enhance(sharp)

    # Convert to OpenCV
    cv_img = np.array(img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Denoise
    denoised = cv2.fastNlMeansDenoisingColored(cv_img, None, denoise_h, denoise_h, 7, 21)

    # Convert back to PIL for display/save
    result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)


def main():
    st.title("🖼️ Image Enhancement App")
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Original Image", use_column_width=True)
        
        # Sidebar controls for parameters
        st.sidebar.header("Enhancement Parameters")
        bright = st.sidebar.slider("Brightness", 0.5, 3.0, 1.2, 0.1)
        contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.3, 0.1)
        sharp = st.sidebar.slider("Sharpness", 0.0, 5.0, 2.0, 0.1)
        denoise_h = st.sidebar.slider("Denoise Strength", 0, 30, 10, 1)
        
        if st.button("Enhance Image"):
            with st.spinner("Enhancing..."):
                enhanced = enhance_image(pil_img, bright, contrast, sharp, denoise_h)
                st.image(enhanced, caption="Enhanced Image", use_column_width=True)
                
                # Save to temporary file and provide download
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                enhanced.save(tmp_file.name, format="JPEG")
                
                st.success("✅ Enhanced image ready!")
                st.download_button(
                    label="Download Enhanced Image",
                    data=open(tmp_file.name, "rb").read(),
                    file_name="enhanced_image.jpg",
                    mime="image/jpeg"
                )

if __name__ == "__main__":
    main()
