import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from pathlib import Path
import argparse
import time
import threading
import queue
from matplotlib.animation import FuncAnimation

warnings.filterwarnings('ignore')

# Configuration (should match your training config)
class TestConfig:
    img_size = 224
    num_classes = 8
    dropout_rate = 0.3
    model_name = 'convnext_base'
    model_path = 'C:\\Users\\harsh\\OneDrive\\Desktop\\MajorProject\\backend\\best_model.pth'  # Update this path
    
config = TestConfig()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Emotion names (should match your training order)
emotion_names = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutrality', 'sadness', 'surprise']

# SOTA Model Architecture (same as training)
class SOTAEmotionModel(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=8, dropout_rate=0.3):
        super(SOTAEmotionModel, self).__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,
            drop_rate=dropout_rate,
            drop_path_rate=0.2
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            self.feature_dim = dummy_features.shape[1]
        
        # Squeeze-and-Excitation Attention
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim // 16),
            nn.SiLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim, 
            num_heads=16, 
            dropout=0.1,
            batch_first=True
        )
        
        # Advanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.7),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply SE attention
        se_weights = self.se_block(features.unsqueeze(-1))
        features_se = features * se_weights
        
        # Self-attention
        features_reshaped = features_se.unsqueeze(1)
        attended_features, _ = self.attention(
            features_reshaped, features_reshaped, features_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        # Combine features
        combined_features = features + features_se + attended_features
        
        # Classification
        output = self.classifier(combined_features)
        
        return output

class EmotionPredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.emotion_names = emotion_names
        
        # Load model
        print("Loading trained model...")
        self.model = SOTAEmotionModel(
            config.model_name, 
            config.num_classes, 
            config.dropout_rate
        ).to(device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✅ Model loaded successfully from {model_path}")
        
        # Setup preprocessing transforms (same as validation transforms)
        self.transform = A.Compose([
            A.Resize(config.img_size, config.img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def enhance_image(self, image):
        """Apply the same image enhancement as during training"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)
        
        # Noise reduction
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Convert back to PIL
        image = Image.fromarray(img_array)
        
        return image
    
    def preprocess_image(self, image_path_or_array):
        """Preprocess image for model input"""
        # Handle different input types
        if isinstance(image_path_or_array, str):
            # Load from file path
            image = Image.open(image_path_or_array).convert('L')
        elif isinstance(image_path_or_array, np.ndarray):
            # Convert numpy array to PIL
            if len(image_path_or_array.shape) == 3:
                # Convert to grayscale if RGB
                image = cv2.cvtColor(image_path_or_array, cv2.COLOR_RGB2GRAY)
            else:
                image = image_path_or_array
            image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image_path_or_array, Image.Image):
            # Already PIL Image
            image = image_path_or_array.convert('L')
        else:
            raise ValueError("Input must be image path, numpy array, or PIL Image")
        
        # Apply image enhancement
        image = self.enhance_image(image)
        
        # Convert grayscale to RGB
        image = image.convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # Add batch dimension
        image = image.unsqueeze(0)
        
        return image
    
    def predict_emotion(self, image_input, return_probabilities=False):
        """Predict emotion from image"""
        # Preprocess image
        processed_image = self.preprocess_image(image_input)
        processed_image = processed_image.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(processed_image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_emotion = self.emotion_names[predicted.item()]
            confidence_score = confidence.item()
        
        if return_probabilities:
            probs_dict = {
                emotion: prob.item() 
                for emotion, prob in zip(self.emotion_names, probabilities[0])
            }
            return predicted_emotion, confidence_score, probs_dict
        
        return predicted_emotion, confidence_score

# WebCam Frame Capture Thread
class WebCamCapture:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.frame_queue = queue.Queue(maxsize=5)
        self.running = False
        self.thread = None
        
    def start(self):
        if not self.cap.isOpened():
            raise ValueError("Could not open camera")
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        print("🎥 Camera capture started")
        
    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Keep only the latest frame
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()  # Remove old frame
                        self.frame_queue.put(frame)
                    except queue.Empty:
                        pass
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.cap.release()
        print("🛑 Camera stopped")

def test_single_image(predictor, image_path, show_probabilities=False):
    """Test single image and display results"""
    print(f"\n{'='*60}")
    print(f"TESTING IMAGE: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    try:
        # Predict emotion
        start_time = time.time()
        if show_probabilities:
            emotion, confidence, probabilities = predictor.predict_emotion(
                image_path, return_probabilities=True
            )
        else:
            emotion, confidence = predictor.predict_emotion(image_path)
        
        inference_time = time.time() - start_time
        
        # Display results
        print(f"🎭 Predicted Emotion: {emotion.upper()}")
        print(f"📊 Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"⏱️  Inference Time: {inference_time:.4f} seconds")
        
        if show_probabilities:
            print(f"\n📈 All Emotion Probabilities:")
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            for emotion_name, prob in sorted_probs:
                bar = "█" * int(prob * 20)  # Visual bar
                print(f"   {emotion_name:12}: {prob:.4f} ({prob*100:5.2f}%) {bar}")
        
        # Load and display image
        try:
            img = Image.open(image_path)
            plt.figure(figsize=(8, 6))
            plt.imshow(img, cmap='gray' if img.mode == 'L' else None)
            plt.title(f'Predicted: {emotion.upper()} (Confidence: {confidence:.2f})', 
                     fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not display image: {e}")
            
        return emotion, confidence
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        return None, None

def test_webcam_matplotlib_live(predictor):
    """Real-time webcam testing using matplotlib (Windows compatible)"""
    print(f"\n{'='*60}")
    print("🎥 STARTING MATPLOTLIB LIVE WEBCAM TESTING")
    print("Close the window or press Ctrl+C to stop")
    print(f"{'='*60}")
    
    # Initialize camera capture
    try:
        camera = WebCamCapture(0)
        camera.start()
        time.sleep(2)  # Let camera warm up
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        return
    
    # Setup matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Real-time Emotion Recognition', fontsize=16, fontweight='bold')
    
    ax1.set_title('Live Camera Feed')
    ax1.axis('off')
    
    ax2.set_title('Emotion Probabilities')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Confidence')
    ax2.tick_params(axis='x', rotation=45)
    
    # Initialize plots
    im1 = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    bars = ax2.bar(range(len(emotion_names)), [0] * len(emotion_names))
    ax2.set_xticks(range(len(emotion_names)))
    ax2.set_xticklabels([name.capitalize() for name in emotion_names])
    
    # Prediction variables
    current_emotion = "Initializing..."
    current_confidence = 0.0
    frame_count = 0
    last_prediction_time = 0
    
    def update_frame(frame_num):
        nonlocal current_emotion, current_confidence, frame_count, last_prediction_time
        
        # Get latest frame
        frame = camera.get_frame()
        if frame is None:
            return [im1] + list(bars)
        
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update camera feed
        im1.set_array(rgb_frame)
        
        # Make prediction every 10 frames (for performance)
        if frame_count % 10 == 0:
            try:
                start_time = time.time()
                emotion, confidence, probabilities = predictor.predict_emotion(
                    rgb_frame, return_probabilities=True
                )
                
                current_emotion = emotion
                current_confidence = confidence
                last_prediction_time = time.time() - start_time
                
                # Update probability bars
                for i, (bar, prob) in enumerate(zip(bars, probabilities.values())):
                    bar.set_height(prob)
                    # Highlight the predicted emotion
                    if emotion_names[i] == emotion:
                        bar.set_color('red')
                    else:
                        bar.set_color('skyblue')
                
                print(f"📊 Frame {frame_count:4d} | Emotion: {emotion:12} | Confidence: {confidence:.3f} | Time: {last_prediction_time:.3f}s")
                
            except Exception as e:
                print(f"❌ Prediction error: {e}")
        
        # Update title with current prediction
        ax1.set_title(f'Live Feed | Emotion: {current_emotion.upper()} | Confidence: {current_confidence:.3f}')
        
        return [im1] + list(bars)
    
    # Start animation
    try:
        ani = FuncAnimation(fig, update_frame, interval=50, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Display error: {e}")
    finally:
        camera.stop()
        plt.close('all')
        print("✅ Session ended")

def test_webcam_console_only(predictor):
    """Console-only webcam testing (always works)"""
    print(f"\n{'='*60}")
    print("📱 CONSOLE-ONLY WEBCAM EMOTION RECOGNITION")
    print("Press Ctrl+C to stop")
    print(f"{'='*60}")
    
    # Initialize camera
    try:
        camera = WebCamCapture(0)
        camera.start()
        time.sleep(2)  # Let camera warm up
        print("🎥 Camera initialized successfully!")
    except Exception as e:
        print(f"❌ Camera initialization failed: {e}")
        return
    
    frame_count = 0
    prediction_count = 0
    total_inference_time = 0
    
    try:
        print("\n🔄 Starting emotion recognition...")
        print(f"{'Frame':>6} | {'Emotion':>12} | {'Confidence':>10} | {'Time(ms)':>9}")
        print("-" * 50)
        
        while True:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Process every 15th frame for better performance
            if frame_count % 15 == 0:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    start_time = time.time()
                    emotion, confidence = predictor.predict_emotion(rgb_frame)
                    inference_time = time.time() - start_time
                    
                    prediction_count += 1
                    total_inference_time += inference_time
                    
                    print(f"{frame_count:6d} | {emotion:>12} | {confidence:>10.3f} | {inference_time*1000:>7.1f}ms")
                    
                    # Save a frame every 50 predictions
                    if prediction_count % 50 == 0:
                        timestamp = int(time.time())
                        filename = f"emotion_capture_{emotion}_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        print(f"         💾 Saved: {filename}")
                        
                        # Show average performance
                        avg_time = (total_inference_time / prediction_count) * 1000
                        print(f"         📊 Average inference time: {avg_time:.1f}ms")
                
                except Exception as e:
                    print(f"         ❌ Error: {str(e)[:30]}")
            
            time.sleep(0.01)  # Small delay
    
    except KeyboardInterrupt:
        print(f"\n🛑 Session ended by user")
        if prediction_count > 0:
            avg_time = (total_inference_time / prediction_count) * 1000
            print(f"\n📊 FINAL STATISTICS:")
            print(f"   Total frames processed: {frame_count}")
            print(f"   Total predictions: {prediction_count}")
            print(f"   Average inference time: {avg_time:.1f}ms")
    finally:
        camera.stop()

def test_folder_batch(predictor, folder_path):
    """Test all images in a folder"""
    print(f"\n{'='*60}")
    print(f"BATCH TESTING FOLDER: {folder_path}")
    print(f"{'='*60}")
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    folder_path = Path(folder_path)
    for ext in image_extensions:
        image_files.extend(list(folder_path.glob(f"*{ext}")))
        image_files.extend(list(folder_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"❌ No image files found in {folder_path}")
        return
    
    print(f"📁 Found {len(image_files)} image files")
    
    # Process images
    results = []
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"\nProcessing {i}/{len(image_files)}: {image_file.name}")
            emotion, confidence = predictor.predict_emotion(str(image_file))
            results.append({
                'filename': image_file.name,
                'emotion': emotion,
                'confidence': confidence
            })
            print(f"  → {emotion} ({confidence:.3f})")
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results.append({
                'filename': image_file.name,
                'emotion': 'Error',
                'confidence': 0.0
            })
    
    # Display summary
    print(f"\n{'='*40}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*40}")
    
    emotion_counts = {}
    total_confidence = 0
    successful_predictions = 0
    
    for result in results:
        if result['emotion'] != 'Error':
            emotion_counts[result['emotion']] = emotion_counts.get(result['emotion'], 0) + 1
            total_confidence += result['confidence']
            successful_predictions += 1
    
    print(f"📊 Successful predictions: {successful_predictions}/{len(results)}")
    if successful_predictions > 0:
        print(f"📈 Average confidence: {total_confidence/successful_predictions:.3f}")
        
        print(f"\n🎭 Emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            percentage = (count / successful_predictions) * 100
            print(f"   {emotion:12}: {count:2d} images ({percentage:5.1f}%)")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Real-time Emotion Recognition Tester')
    parser.add_argument('--model', '-m', type=str, default=config.model_path,
                       help='Path to trained model file')
    parser.add_argument('--mode', '-M', type=str, 
                       choices=['image', 'webcam', 'webcam-console', 'folder'], 
                       default='image', help='Testing mode')
    parser.add_argument('--input', '-i', type=str, 
                       help='Input image path or folder path')
    parser.add_argument('--show-probs', '-p', action='store_true',
                       help='Show all emotion probabilities')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("🎭 REAL-TIME EMOTION RECOGNITION TESTER 🎭")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    
    # Initialize predictor
    try:
        predictor = EmotionPredictor(args.model, device)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print("💡 Please check the model path and ensure the model file exists")
        return
    
    # Run different modes
    if args.mode == 'image':
        if not args.input:
            print("❌ Please provide an image path using --input")
            return
        
        if not os.path.exists(args.input):
            print(f"❌ Image file not found: {args.input}")
            return
            
        test_single_image(predictor, args.input, args.show_probs)
    
    elif args.mode == 'webcam':
        print("🔍 Attempting matplotlib live display...")
        try:
            test_webcam_matplotlib_live(predictor)
        except Exception as e:
            print(f"⚠️  Matplotlib display failed: {e}")
            print("🔄 Falling back to console mode...")
            test_webcam_console_only(predictor)
    
    elif args.mode == 'webcam-console':
        test_webcam_console_only(predictor)
    
    elif args.mode == 'folder':
        if not args.input:
            print("❌ Please provide a folder path using --input")
            return
            
        if not os.path.exists(args.input):
            print(f"❌ Folder not found: {args.input}")
            return
            
        test_folder_batch(predictor, args.input)

# Quick test functions for direct use
def quick_test_image(image_path, model_path=None):
    """Quick function to test a single image"""
    if model_path is None:
        model_path = config.model_path
    
    predictor = EmotionPredictor(model_path, device)
    return test_single_image(predictor, image_path, show_probabilities=True)

def quick_test_webcam_live(model_path=None):
    """Quick function to start live webcam testing"""
    if model_path is None:
        model_path = config.model_path
    
    predictor = EmotionPredictor(model_path, device)
    
    print("🔍 Trying matplotlib live display...")
    try:
        test_webcam_matplotlib_live(predictor)
    except Exception as e:
        print(f"⚠️  Live display failed: {e}")
        print("🔄 Using console mode...")
        test_webcam_console_only(predictor)

def quick_test_webcam_console(model_path=None):
    """Quick function for console-only webcam testing"""
    if model_path is None:
        model_path = config.model_path
    
    predictor = EmotionPredictor(model_path, device)
    test_webcam_console_only(predictor)

# Fix for OpenCV installation
def fix_opencv_installation():
    """Instructions to fix OpenCV GUI issues on Windows"""
    print(f"\n{'='*80}")
    print("🔧 OPENCV GUI INSTALLATION FIX")
    print(f"{'='*80}")
    print("Your OpenCV installation is missing GUI components. To fix this:")
    print("\n1. Uninstall current OpenCV:")
    print("   pip uninstall opencv-python opencv-contrib-python")
    print("\n2. Install the headless version (recommended for servers):")
    print("   pip install opencv-python-headless")
    print("\n3. OR install the full version with GUI support:")
    print("   pip install opencv-contrib-python")
    print("\n4. For conda users:")
    print("   conda install -c conda-forge opencv")
    print(f"\n{'='*80}")
    print("💡 Meanwhile, use the console mode: python test.py --mode webcam-console")
    print(f"{'='*80}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if "function is not implemented" in str(e).lower():
            fix_opencv_installation()
        else:
            print(f"❌ Error: {e}")

# USAGE EXAMPLES:
"""
🚀 FIXED USAGE EXAMPLES:

Command Line:
python test.py --mode image --input path/to/image.jpg --show-probs
python test.py --mode webcam                    # Matplotlib live display
python test.py --mode webcam-console            # Console only (always works)
python test.py --mode folder --input path/to/images/

In Python/Jupyter:
quick_test_image('path/to/image.jpg')
quick_test_webcam_live()         # Matplotlib display
quick_test_webcam_console()      # Console only
"""