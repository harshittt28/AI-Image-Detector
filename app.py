import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm

# -----------------------------------
# Custom CSS for Beautiful Gradient Design
# -----------------------------------
st.set_page_config(page_title="AI Image Detector - All Models", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: #000000 !important;
    }
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
    }
    .title-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    .title-box h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 36px !important;
    }
    .title-box p {
        color: white !important;
        margin: 10px 0 0 0 !important;
        font-size: 18px !important;
    }
    .upload-section {
        background: transparent !important;
        padding: 0px !important;
        margin-bottom: 20px;
    }
    .results-container {
        background: #0a0a0a;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(255, 255, 255, 0.05);
        border: 1px solid #1a1a1a;
    }
    .image-display {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #2c3e50 0%, #1a1a1a 100%);
        border-radius: 15px;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.5);
    }
    .image-wrapper {
        display: inline-block;
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 8px;
        background: #ffffff;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.5);
    }
    .image-wrapper img {
        display: block;
        border-radius: 8px;
    }
    .predictions-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        font-weight: bold;
        font-size: 22px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }
    .model-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 22px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(42, 82, 152, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .model-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .model-name {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 3px solid rgba(255, 255, 255, 0.3);
    }
    .prediction-real {
        color: #00ff88;
        font-weight: bold;
        font-size: 18px;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    .prediction-ai {
        color: #ff4757;
        font-weight: bold;
        font-size: 18px;
        text-shadow: 0 0 10px rgba(255, 71, 87, 0.5);
    }
    .confidence-value {
        font-size: 32px;
        font-weight: bold;
        color: #ffffff;
        margin: 12px 0;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.3);
    }
    progress {
        width: 100%;
        height: 26px;
        border-radius: 13px;
        margin-top: 12px;
    }
    progress::-webkit-progress-bar {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 13px;
    }
    progress::-webkit-progress-value {
        background: linear-gradient(90deg, #00ff88 0%, #00d9ff 100%);
        border-radius: 13px;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.5);
    }
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 70px 50px;
        border-radius: 20px;
        text-align: center;
        margin: 50px auto;
        max-width: 900px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    .info-box h2 {
        color: #ffffff !important;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .info-box p {
        color: #e0e0e0 !important;
        font-size: 18px;
    }
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploader"] section {
        background: #0a0a0a !important;
        border: 2px dashed #667eea !important;
        border-radius: 15px !important;
        padding: 20px !important;
    }
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 18px;
    }
    [data-testid="stFileUploader"] small {
        color: #aaaaaa !important;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
    }
    [data-testid="stMultiSelect"] {
        background: #0a0a0a;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #667eea;
        margin-bottom: 20px;
    }
    [data-testid="stMultiSelect"] label {
        color: #ffffff !important;
        font-weight: bold;
        font-size: 16px;
    }
    [data-testid="stMultiSelect"] > div {
        background: #1a1a1a !important;
        border: 1px solid #667eea !important;
        border-radius: 8px !important;
    }
    .stMarkdown h3 {
        color: #ffffff !important;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load All Models
# -----------------------------------
@st.cache_resource
def load_models():
    models = {}
    
    # Load TensorFlow/Keras models
    try:
        models['VGG16'] = tf.keras.models.load_model("vgg16.h5", compile=False)
        print("VGG16 loaded successfully")
    except Exception as e:
        models['VGG16'] = None
        print(f"VGG16 loading error: {str(e)}")
    
    try:
        models['ResNet50'] = tf.keras.models.load_model("resnet50.h5", compile=False)
        print("ResNet50 loaded successfully")
    except Exception as e:
        models['ResNet50'] = None
        print(f"ResNet50 loading error: {str(e)}")
    
    try:
        models['MobileNetV2'] = tf.keras.models.load_model("mobilenetv2.h5", compile=False)
        print("MobileNetV2 loaded successfully")
    except Exception as e:
        models['MobileNetV2'] = None
        print(f"MobileNetV2 loading error: {str(e)}")
    
    try:
        models['DenseNet121'] = tf.keras.models.load_model("densenet121.h5", compile=False)
        print("DenseNet121 loaded successfully")
    except Exception as e:
        models['DenseNet121'] = None
        print(f"DenseNet121 loading error: {str(e)}")
    
    try:
        models['InceptionV3'] = tf.keras.models.load_model("inceptionv3.h5", compile=False)
        print("InceptionV3 loaded successfully")
    except Exception as e:
        models['InceptionV3'] = None
        print(f"InceptionV3 loading error: {str(e)}")
    
    # Load PyTorch EfficientNet model
    try:
        device = torch.device('cpu')
        efficientnet_model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2)
        efficientnet_model.load_state_dict(torch.load("efficientnet_b0.pth", map_location=device))
        efficientnet_model.eval()
        models['EfficientNetB0'] = efficientnet_model
        print("EfficientNetB0 loaded successfully")
    except Exception as e:
        models['EfficientNetB0'] = None
        print(f"EfficientNetB0 loading error: {str(e)}")
    
    return models

models = load_models()

# -----------------------------------
# Preprocessing Functions
# -----------------------------------
def preprocess_vgg16(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_resnet50(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_mobilenetv2(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_densenet121(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.densenet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_inceptionv3(img):
    img = img.resize((224, 224))  # Using 224 as per your trained model
    img = np.array(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_efficientnet(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

# -----------------------------------
# Helper function to resize image with aspect ratio
# -----------------------------------
def resize_with_aspect_ratio(image, max_width=400, max_height=400):
    img_width, img_height = image.size
    aspect_ratio = img_width / img_height
    
    if img_width > img_height:
        new_width = min(max_width, img_width)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(max_height, img_height)
        new_width = int(new_height * aspect_ratio)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

# -----------------------------------
# Prediction Functions
# -----------------------------------
def predict_all_models(img):
    results = {}
    
    if models['VGG16'] is not None:
        try:
            processed = preprocess_vgg16(img)
            pred = models['VGG16'].predict(processed, verbose=0)[0][0]
            results['VGG16'] = float(pred)
        except:
            results['VGG16'] = None
    
    if models['ResNet50'] is not None:
        try:
            processed = preprocess_resnet50(img)
            pred = models['ResNet50'].predict(processed, verbose=0)[0][0]
            results['ResNet50'] = float(pred)
        except:
            results['ResNet50'] = None
    
    if models['MobileNetV2'] is not None:
        try:
            processed = preprocess_mobilenetv2(img)
            pred = models['MobileNetV2'].predict(processed, verbose=0)[0][0]
            results['MobileNetV2'] = float(pred)
        except:
            results['MobileNetV2'] = None
    
    if models['DenseNet121'] is not None:
        try:
            processed = preprocess_densenet121(img)
            pred = models['DenseNet121'].predict(processed, verbose=0)[0][0]
            results['DenseNet121'] = float(pred)
        except:
            results['DenseNet121'] = None
    
    if models['InceptionV3'] is not None:
        try:
            processed = preprocess_inceptionv3(img)
            pred = models['InceptionV3'].predict(processed, verbose=0)[0][0]
            results['InceptionV3'] = float(pred)
        except Exception as e:
            results['InceptionV3'] = None
    
    if models['EfficientNetB0'] is not None:
        try:
            processed = preprocess_efficientnet(img)
            with torch.no_grad():
                pred = models['EfficientNetB0'](processed)
                # For binary classification, get probability of class 1 (Real)
                pred = torch.softmax(pred, dim=1)[0][1].item()
            results['EfficientNetB0'] = float(pred)
        except Exception as e:
            results['EfficientNetB0'] = None
            print(f"EfficientNet prediction error: {str(e)}")
    
    return results

# -----------------------------------
# UI Layout
# -----------------------------------
st.markdown("""
    <div class="title-box">
        <h1 style='color:white; margin:0; font-size:36px;'>AI Generated Image Detector</h1>
        <p style='color:white; font-size:18px; margin:10px 0 0 0;'>Multi-Model Analysis Dashboard</p>
    </div>
""", unsafe_allow_html=True)

# Upload and Model Selection in same row
col_upload, col_select = st.columns([1, 1])

with col_upload:
    uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

with col_select:
    st.markdown("### Select Models")
    all_models = ['VGG16', 'ResNet50', 'MobileNetV2', 'DenseNet121', 'InceptionV3', 'EfficientNetB0']
    selected_models = st.multiselect(
        "Choose which models to analyze",
        all_models,
        default=all_models,
        label_visibility="collapsed"
    )

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    
    if not selected_models:
        st.warning("Please select at least one model to analyze!")
    else:
        # Results container
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        
        # Create two column layout - Image on left, predictions on right
        col_img, col_pred = st.columns([1, 2], gap="large")
        
        with col_img:
            # Display image
            st.markdown('<div class="image-display">', unsafe_allow_html=True)
            
            # Resize image with aspect ratio
            display_img = resize_with_aspect_ratio(img, max_width=450, max_height=500)
            
            # Display image with frame using HTML
            import io
            import base64
            buffered = io.BytesIO()
            display_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            st.markdown(f'''
                <div class="image-wrapper">
                    <img src="data:image/png;base64,{img_str}" style="max-width: 450px; max-height: 500px; border-radius: 6px;">
                </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_pred:
            # Model predictions section
            st.markdown('<div class="predictions-header">Model Predictions</div>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing with selected models..."):
                results = predict_all_models(img)
            
            # Filter results based on selected models
            filtered_results = {k: v for k, v in results.items() if k in selected_models}
            
            # Dynamic grid based on number of selected models
            num_models = len(selected_models)
            if num_models == 1:
                cols = [st.container()]
            elif num_models == 2:
                cols = list(st.columns(2))
            elif num_models <= 4:
                row1_cols = st.columns(2)
                row2_cols = st.columns(2) if num_models > 2 else []
                cols = list(row1_cols) + list(row2_cols)
            else:
                row1_cols = st.columns(2)
                row2_cols = st.columns(2)
                row3_cols = st.columns(2)
                cols = list(row1_cols) + list(row2_cols) + list(row3_cols)
            
            for idx, model_name in enumerate(selected_models):
                with cols[idx]:
                    if filtered_results.get(model_name) is not None:
                        confidence = filtered_results[model_name]
                        
                        if confidence >= 0.5:
                            label = "Real"
                            prob = confidence * 100
                            pred_class = "prediction-real"
                        else:
                            label = "AI-Generated"
                            prob = (1 - confidence) * 100
                            pred_class = "prediction-ai"
                        
                        st.markdown(f"""
                            <div class="model-card">
                                <div class="model-name">{model_name}</div>
                                <div class="{pred_class}">{label}</div>
                                <div class="confidence-value">{prob:.1f}%</div>
                                <progress value="{prob}" max="100"></progress>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="model-card">
                                <div class="model-name">{model_name}</div>
                                <div style="color:#e67e22; font-size:14px; margin-top:10px;">Loading Error</div>
                                <div style="color:#95a5a6; font-size:12px; margin-top:5px;">Check console</div>
                            </div>
                        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
        <div class="info-box">
            <h2 style='color:#667eea; margin-bottom:20px; font-size:32px;'>Welcome to AI Image Detector</h2>
            <p style='color:#666; font-size:18px; margin-bottom:25px;'>Upload an image to analyze with 6 different AI models</p>
            <p style='color:#999; font-size:16px;'>Supported formats: JPG, JPEG, PNG</p>
        </div>
    """, unsafe_allow_html=True)
