"""
DEEPFAKE DETECTION STREAMLIT UI
Complete interface for training and testing deepfake detection models

Installation:
pip install streamlit tensorflow opencv-python numpy matplotlib pillow pandas plotly
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🔍 Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL BUILDING FUNCTIONS
# ============================================================================

@st.cache_resource
def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """Build transfer learning model with EfficientNet"""
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

@st.cache_resource
def build_simple_cnn(input_shape=(224, 224, 3)):
    """Build simple CNN model"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def create_data_generators(dataset_path, img_size=(224, 224), batch_size=32):
    """Create data generators for training"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(dataset_path, 'Train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['Real', 'Fake'],
        shuffle=True
    )
    
    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'Validation'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['Real', 'Fake'],
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(dataset_path, 'Test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        classes=['Real', 'Fake'],
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator

# ============================================================================
# PREDICTION CLASS
# ============================================================================

class DeepfakeDetector:
    """Deepfake detection class"""
    
    def __init__(self, model_path, img_size=(224, 224)):
        self.model = keras.models.load_model(model_path)
        self.img_size = img_size
    
    def predict_image(self, image, threshold=0.5):
        """Predict if image is fake or real"""
        # Preprocess
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(image)
        
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = self.model.predict(img, verbose=0)[0][0]
        
        if prediction >= threshold:
            label = 'FAKE'
            confidence = prediction * 100
        else:
            label = 'REAL'
            confidence = (1 - prediction) * 100
        
        return label, confidence, prediction

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_metrics(history):
    """Create interactive training plots"""
    fig = go.Figure()
    
    # Accuracy
    fig.add_trace(go.Scatter(
        y=history.history['accuracy'],
        name='Train Accuracy',
        mode='lines+markers',
        line=dict(color='#667eea', width=3)
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_accuracy'],
        name='Val Accuracy',
        mode='lines+markers',
        line=dict(color='#764ba2', width=3)
    ))
    
    fig.update_layout(
        title='Training & Validation Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_prediction_gauge(confidence, label):
    """Create gauge chart for prediction confidence"""
    color = '#ef4444' if label == 'FAKE' else '#10b981'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Prediction: {label}"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# ============================================================================
# STREAMLIT APP PAGES
# ============================================================================

def home_page():
    """Home page"""
    st.markdown('<h1 class="main-header">🔍 Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Image Authenticity Verification</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>🎯</h2>
            <h3>Accurate</h3>
            <p>State-of-the-art CNN models</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>⚡</h2>
            <h3>Fast</h3>
            <p>Real-time predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>🛡️</h2>
            <h3>Reliable</h3>
            <p>Robust detection algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📋 Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 🎓 **Train Custom Models** - Train on your own dataset
        - 📊 **Real-time Monitoring** - Track training progress
        - 📈 **Performance Metrics** - Comprehensive evaluation
        """)
    
    with col2:
        st.markdown("""
        - 🖼️ **Image Prediction** - Upload and analyze images
        - 📁 **Batch Processing** - Analyze multiple images
        - 💾 **Model Management** - Save and load models
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Quick Start Guide")
    st.markdown("""
    1. **Train a Model**: Go to 'Train Model' page and configure training parameters
    2. **Make Predictions**: Upload images in 'Predict' page to detect deepfakes
    3. **View Results**: Analyze confidence scores and batch results
    4. **Download Model**: Save trained models for future use
    """)

def train_page():
    """Training page"""
    st.header("🎓 Train Deepfake Detection Model")
    
    with st.expander("ℹ️ Training Information", expanded=True):
        st.markdown("""
        - **Dataset Structure**: Organize your dataset with Train/Validation/Test folders
        - Each folder should contain 'Real' and 'Fake' subfolders
        - Recommended: At least 500 images per class for good results
        - **Model Types**: 
          - Transfer Learning (EfficientNet) - Better accuracy, more memory
          - Simple CNN - Faster training, less memory
        """)
    
    st.markdown("---")
    
    # Training Configuration
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dataset_path = st.text_input(
            "📁 Dataset Path",
            value=r"C:\Users\adarsh\OneDrive\Documents\GitHub\DEEP_FAKE_DETECTION\data_set",
            help="Path to your dataset folder"
        )
        
        model_type = st.selectbox(
            "🤖 Model Type",
            ["Transfer Learning (EfficientNet)", "Simple CNN"],
            help="Choose between transfer learning (better accuracy) or simple CNN (faster)"
        )
        
        img_size = st.selectbox(
            "📏 Image Size",
            [128, 224, 256],
            index=1,
            help="Input image size - larger = better quality but slower"
        )
    
    with col2:
        batch_size = st.slider(
            "📦 Batch Size",
            min_value=8,
            max_value=64,
            value=32,
            step=8,
            help="Number of images per batch - reduce if out of memory"
        )
        
        epochs = st.slider(
            "🔄 Number of Epochs",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Training iterations - more epochs = better learning but longer training"
        )
        
        learning_rate = st.select_slider(
            "📊 Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            help="Learning rate for optimizer"
        )
    
    model_name = st.text_input(
        "💾 Model Save Name",
        value="deepfake_model.h5",
        help="Filename for saving the trained model"
    )
    
    st.markdown("---")
    
    # Training Button
    if st.button("🚀 Start Training", key="train_btn"):
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            st.error("❌ Dataset path does not exist!")
            return
        
        # Create progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.container()
        
        try:
            # Load data
            status_text.text("📁 Loading dataset...")
            train_gen, val_gen, test_gen = create_data_generators(
                dataset_path,
                img_size=(img_size, img_size),
                batch_size=batch_size
            )
            
            st.success(f"✅ Dataset loaded: {train_gen.samples} training images")
            
            # Build model
            status_text.text("🏗️ Building model...")
            if "Transfer" in model_type:
                model = build_transfer_learning_model(input_shape=(img_size, img_size, 3))
            else:
                model = build_simple_cnn(input_shape=(img_size, img_size, 3))
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy',
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')]
            )
            
            st.success(f"✅ Model built: {model.count_params():,} parameters")
            
            # Training callbacks
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    model_name,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            # Train model with progress updates
            status_text.text("🎓 Training model...")
            
            # Create metrics display
            with metrics_container:
                metric_cols = st.columns(4)
                acc_metric = metric_cols[0].empty()
                loss_metric = metric_cols[1].empty()
                val_acc_metric = metric_cols[2].empty()
                val_loss_metric = metric_cols[3].empty()
            
            # Custom callback for progress
            class StreamlitCallback(keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    status_text.text(f"Epoch {epoch + 1}/{epochs}")
                    
                    # Update metrics
                    acc_metric.metric("Train Accuracy", f"{logs['accuracy']:.4f}")
                    loss_metric.metric("Train Loss", f"{logs['loss']:.4f}")
                    val_acc_metric.metric("Val Accuracy", f"{logs['val_accuracy']:.4f}")
                    val_loss_metric.metric("Val Loss", f"{logs['val_loss']:.4f}")
            
            callbacks.append(StreamlitCallback())
            
            # Train
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Training complete!")
            
            # Evaluate on test set
            st.subheader("📊 Test Set Evaluation")
            test_results = model.evaluate(test_gen, verbose=0)
            
            result_cols = st.columns(5)
            result_cols[0].metric("Loss", f"{test_results[0]:.4f}")
            result_cols[1].metric("Accuracy", f"{test_results[1]*100:.2f}%")
            result_cols[2].metric("Precision", f"{test_results[2]:.4f}")
            result_cols[3].metric("Recall", f"{test_results[3]:.4f}")
            result_cols[4].metric("AUC", f"{test_results[4]:.4f}")
            
            # Plot training history
            st.subheader("📈 Training History")
            fig = plot_training_metrics(history)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"🎉 Model saved as: {model_name}")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.exception(e)

def predict_page():
    """Prediction page"""
    st.header("🔮 Deepfake Detection Prediction")
    
    # Model selection
    st.subheader("📂 Load Model")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_path = st.text_input(
            "Model Path",
            value="best_deepfake_model.h5",
            help="Path to your trained model file"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        load_model = st.button("📥 Load Model")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.warning("⚠️ Model file not found. Please train a model first or provide correct path.")
        return
    
    # Load detector
    try:
        detector = DeepfakeDetector(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return
    
    st.markdown("---")
    
    # Tabs for different prediction modes
    tab1, tab2 = st.tabs(["📸 Single Image", "📁 Batch Processing"])
    
    with tab1:
        st.subheader("Upload Image for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image to check if it's real or fake"
        )
        
        threshold = st.slider(
            "🎯 Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Classification threshold - higher = more conservative"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Make prediction
                with st.spinner("🔍 Analyzing image..."):
                    label, confidence, raw_score = detector.predict_image(image, threshold)
                
                # Display results
                st.markdown("### 🎯 Detection Results")
                
                # Gauge chart
                fig = create_prediction_gauge(confidence, label)
                st.plotly_chart(fig, use_container_width=True)
                
                # Result card
                result_color = "#ef4444" if label == "FAKE" else "#10b981"
                st.markdown(f"""
                <div style="background: {result_color}; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                    <h2>Classification: {label}</h2>
                    <h3>Confidence: {confidence:.2f}%</h3>
                    <p>Raw Score: {raw_score:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Interpretation
                st.markdown("### 📊 Interpretation")
                if confidence > 90:
                    st.success(f"🟢 Very high confidence that the image is {label}")
                elif confidence > 70:
                    st.info(f"🟡 High confidence that the image is {label}")
                elif confidence > 50:
                    st.warning(f"🟠 Moderate confidence that the image is {label}")
                else:
                    st.error(f"🔴 Low confidence - result uncertain")
    
    with tab2:
        st.subheader("Batch Image Processing")
        
        uploaded_files = st.file_uploader(
            "Choose multiple images...",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload multiple images for batch processing"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process All Images"):
                results = []
                progress_bar = st.progress(0)
                
                for idx, file in enumerate(uploaded_files):
                    image = Image.open(file)
                    label, confidence, raw_score = detector.predict_image(image)
                    
                    results.append({
                        'Filename': file.name,
                        'Prediction': label,
                        'Confidence (%)': f"{confidence:.2f}",
                        'Raw Score': f"{raw_score:.4f}"
                    })
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Display results table
                st.markdown("### 📊 Batch Results")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                # Summary statistics
                fake_count = sum(1 for r in results if r['Prediction'] == 'FAKE')
                real_count = len(results) - fake_count
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Images", len(results))
                col2.metric("Real Images", real_count, delta_color="normal")
                col3.metric("Fake Images", fake_count, delta_color="inverse")
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="deepfake_detection_results.csv",
                    mime="text/csv"
                )
                
                # Visualization
                fig = px.pie(
                    values=[real_count, fake_count],
                    names=['Real', 'Fake'],
                    title='Detection Distribution',
                    color_discrete_sequence=['#10b981', '#ef4444']
                )
                st.plotly_chart(fig, use_container_width=True)

def about_page():
    """About page"""
    st.header("ℹ️ About Deepfake Detection System")
    
    st.markdown("""
    ## 🎯 What is Deepfake Detection?
    
    Deepfake detection is the process of identifying artificially generated or manipulated images and videos 
    created using artificial intelligence. Our system uses state-of-the-art Convolutional Neural Networks (CNNs) 
    to distinguish between real and fake images with high accuracy.
    
    ## 🧠 How It Works
    
    Our system employs advanced deep learning techniques:
    
    1. **Feature Extraction**: The CNN analyzes pixel patterns, textures, and artifacts
    2. **Pattern Recognition**: Identifies subtle inconsistencies invisible to human eyes
    3. **Classification**: Determines if an image is real or synthetically generated
    
    ## 🔧 Technology Stack
    
    - **TensorFlow/Keras**: Deep learning framework
    - **EfficientNet**: State-of-the-art CNN architecture
    - **Streamlit**: Interactive web interface
    - **OpenCV**: Image processing
    
    ## 📊 Model Architecture
    
    ### Transfer Learning Model (EfficientNet)
    - Pre-trained on ImageNet
    - Fine-tuned for deepfake detection
    - ~4.8M parameters
    - Higher accuracy, more memory intensive
    
    ### Simple CNN Model
    - Custom architecture
    - Optimized for speed
    - ~2M parameters
    - Faster training, less memory usage
    
    ## 🎓 Training Tips
    
    1. **Dataset Quality**: Use diverse, high-quality images
    2. **Balance**: Ensure equal real and fake samples
    3. **Augmentation**: Improves generalization
    4. **Validation**: Monitor validation metrics to prevent overfitting
    
    ## 🤝 Contributing
    
    This is an open-source project. Contributions are welcome!
    
    ## ⚠️ Disclaimer
    
    This tool is for educational and research purposes. Always verify important information 
    through multiple sources and methods.
    
    ## 📧 Contact
    
    For questions or feedback, please open an issue on GitHub.
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: 2024  
    **License**: MIT
    """)

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "🎓 Train Model", "🔮 Predict", "ℹ️ About"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Info")
    st.sidebar.info(f"""
    **TensorFlow**: {tf.__version__}  
    **GPU Available**: {"✅" if tf.config.list_physical_devices('GPU') else "❌"}
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### 🛠️ Quick Actions
    - Train new model
    - Load pretrained model
    - Batch process images
    - Export results
    """)
    
    # Page routing
    if page == "🏠 Home":
        home_page()
    elif page == "🎓 Train Model":
        train_page()
    elif page == "🔮 Predict":
        predict_page()
    elif page == "ℹ️ About":
        about_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Made with ❤️ using Streamlit<br>
        © 2024 Deepfake Detection
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()