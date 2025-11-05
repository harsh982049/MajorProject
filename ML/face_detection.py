import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix
import timm
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageFilter
import cv2
# from facenet_pytorch import MTCNN
import warnings
warnings.filterwarnings('ignore')

# Set your Hugging Face token (only if you're using models from Hugging Face)
# os.environ["HUGGINGFACE_TOKEN"] = "your_token_here"  # Uncomment and replace with your token if needed

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
IMAGE_SIZE = 96  # Input image size
BATCH_SIZE = 32
NUM_EPOCHS = 60
PATIENCE = 15  # More patience to avoid early stopping
BASE_LR = 5e-5  # Lower learning rate for stability
WEIGHT_DECAY = 1e-4
CHECKPOINT_DIR = "/kaggle/working/emotion_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Improved augmentation pipeline
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Larger resize for more crop variety
    # transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.02),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
])

# Standard transform for validation/testing
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset for emotion recognition
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None, face_detection=False):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.classes = self.dataset.classes
        self.transform = transform
        self.targets = [s[1] for s in self.dataset.samples]
        self.samples = self.dataset.samples
        self.class_weights = self._calculate_class_weights()
        self.face_detection = face_detection

        # Initialize face detector if needed
        if self.face_detection:
            self.face_detector = MTCNN(image_size=IMAGE_SIZE, margin=20, keep_all=False,
                                      min_face_size=40, thresholds=[0.6, 0.7, 0.7],
                                      factor=0.707, post_process=True, device=DEVICE)

    def _calculate_class_weights(self):
        """Calculate balanced class weights for loss function"""
        counts = Counter(self.targets)
        total = len(self.targets)

        # Using sqrt for weight calculation prevents extreme values
        weights = {}
        min_count = min(counts.values())
        for cls_id, count in counts.items():
            weights[cls_id] = np.sqrt(min_count / count)

        # Normalize weights to be between 1 and ~3
        max_weight = max(weights.values())
        for cls_id in weights:
            weights[cls_id] = 1 + 2 * (weights[cls_id] / max_weight)

        # Convert to tensor format
        weights_list = [weights.get(i, 1.0) for i in range(len(self.classes))]
        print("\nClass weights:")
        for i, (cls_name, weight) in enumerate(zip(self.classes, weights_list)):
            print(f"  {cls_name}: {weight:.3f}")
        return torch.tensor(weights_list, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_path, label = self.dataset.samples[idx]

        # Load image
        img = Image.open(img_path).convert('RGB')

        # Apply face detection if enabled
        if self.face_detection:
            # Convert PIL to cv2 format
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            # Detect face
            try:
                boxes, _ = self.face_detector.detect(img)
                if boxes is not None:
                    # Get the first face
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    # Add margin
                    h, w = img_cv.shape[:2]
                    margin = int(((x2-x1) + (y2-y1)) / 8)
                    x1 = max(0, x1 - margin)
                    y1 = max(0, y1 - margin)
                    x2 = min(w, x2 + margin)
                    y2 = min(h, y2 + margin)
                    # Crop face
                    img_cv = img_cv[y1:y2, x1:x2]
                    # Convert back to PIL
                    img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            except Exception as e:
                # If face detection fails, use the original image
                pass

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label

# Define paths to your data
TRAIN_PATH = "/kaggle/input/affectnet-shifted/AffectNet_Data/archive (3)/Train"
TEST_PATH = "/kaggle/input/affectnet-shifted/AffectNet_Data/archive (3)/Test"

# Create datasets
train_dataset = EmotionDataset(TRAIN_PATH, train_transform, face_detection=False)
test_dataset = EmotionDataset(TEST_PATH, val_transform, face_detection=False)

print(f"Training samples detected: {len(train_dataset)}")
print(f"Testing samples detected: {len(test_dataset)}")
print("\nEmotion labels:")
for idx, name in enumerate(train_dataset.classes):
    print(f"  {idx}: {name}")

print("\nTraining set class distribution:")
class_counts = Counter(train_dataset.targets)
for idx, name in enumerate(train_dataset.classes):
    print(f"  {name}: {class_counts[idx]} samples")

# Create weighted sampler for balanced training
sample_weights = [train_dataset.class_weights[label].item() for _, label in train_dataset.samples]

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    ),
    num_workers=2,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# Model Architecture: Choose a strong pretrained model
class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes, model_name='vit_base_patch16_224', dropout_rate=0.2):
        super().__init__()

        # Load a pretrained Vision Transformer model (ViT)
        # ViT models are highly effective for facial emotion recognition
        self.base_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # No classification head
            img_size=IMAGE_SIZE,  # Adjust for the image size
        )

        # Get the feature dimension
        self.feature_dim = self.base_model.num_features

        # Custom classifier head with less aggressive regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )

        # Freeze the base model initially
        self._freeze_base_model()

    def _freeze_base_model(self):
        """Freeze all layers in the base model"""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_layers(self, percentage=0.1):
        """Unfreeze a percentage of layers from the end of the base model"""
        # Get all parameters in the base model
        all_params = list(self.base_model.named_parameters())
        num_params = len(all_params)

        # Calculate how many parameters to unfreeze
        num_unfrozen = int(num_params * percentage)

        # Unfreeze the last X% of parameters
        for name, param in all_params[-num_unfrozen:]:
            param.requires_grad = True

        # Count unfrozen parameters
        unfrozen_count = sum(p.requires_grad for p in self.base_model.parameters())
        print(f"Unfrozen {unfrozen_count} parameters in the base model")

    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)

# Choose one of these model architectures - ViT generally performs better
# but is more computationally expensive
model_options = {
    'vit': {
        'name': 'vit_base_patch16_224',
        'dropout': 0.2
    },
    'swin': {
        'name': 'swin_base_patch4_window7_224',
        'dropout': 0.3
    },
    'convnext': {
        'name': 'convnext_base',
        'dropout': 0.4
    },
    'efficient': {
        'name': 'efficientnet_b3',
        'dropout': 0.5
    },
    'resnet': {
        'name': 'resnet50',
        'dropout': 0.5
    }
}

# Choose the model to use (adjust based on your hardware constraints)
selected_model = 'swin'  # Options: 'vit', 'swin', 'convnext', 'efficient', 'resnet'
model_config = model_options[selected_model]

# Create the model
model = EmotionRecognitionModel(
    num_classes=len(train_dataset.classes),
    model_name=model_config['name'],
    dropout_rate=model_config['dropout']
).to(DEVICE)

# Loss function with class weights
criterion = nn.CrossEntropyLoss(
    weight=train_dataset.class_weights.to(DEVICE),
    label_smoothing=0.1  # Label smoothing for better generalization
)

# Configure optimizer with different learning rates for different parts
params = [
    {'params': model.classifier.parameters(), 'lr': BASE_LR},
    {'params': model.base_model.parameters(), 'lr': BASE_LR * 0.1}
]
optimizer = optim.AdamW(params, weight_decay=WEIGHT_DECAY)

# Learning rate scheduler - use OneCycle for faster convergence
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=[BASE_LR, BASE_LR * 0.1],
    steps_per_epoch=len(train_loader),
    epochs=NUM_EPOCHS,
    pct_start=0.2,  # Warm up for 20% of training
    div_factor=25,  # Initial learning rate is max_lr/25
    final_div_factor=1000  # Final learning rate is max_lr/1000
)

# Mixed precision training setup
scaler = GradScaler()

# Mixup augmentation for better generalization
def mixup_data(x, y, alpha=0.2):
    """Apply mixup augmentation to the batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Apply mixup criterion to the predictions"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Training function with mixed precision
def train_epoch(model, loader, optimizer, criterion, scheduler, epoch, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Training")

    for inputs, targets in pbar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        # Apply mixup with probability 0.5
        if random.random() < 0.5:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
            mixed = True
        else:
            mixed = False

        # Mixed precision forward pass
        with autocast():
            outputs = model(inputs)

            if mixed:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                # For accuracy calculation, use the dominant target
                _, preds = torch.max(outputs, 1)
                batch_correct = (lam * (preds == targets_a).float() +
                               (1 - lam) * (preds == targets_b).float()).sum().item()
            else:
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)
                batch_correct = (preds == targets).sum().item()

        # Mixed precision backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Update learning rate
        scheduler.step()

        # Update statistics
        total_loss += loss.item() * inputs.size(0)
        correct += batch_correct
        total += inputs.size(0)

        # Update progress bar
        acc = 100 * correct / total
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

    # Calculate average loss and accuracy for the epoch
    epoch_loss = total_loss / total
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

# Test-time augmentation function
def tta_evaluate(model, loader, num_augmentations=5):
    """Evaluate with test-time augmentation for better results"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    # Create TTA transformations
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip horizontally
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE+8, IMAGE_SIZE+8)),
            transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((IMAGE_SIZE+16, IMAGE_SIZE+16)),
            transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating with TTA"):
            batch_size = inputs.size(0)
            inputs_orig, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Get predictions from original images
            outputs_orig = model(inputs_orig)

            # If TTA is enabled, apply multiple augmentations and average results
            if num_augmentations > 1:
                # Process each image in the batch separately for TTA
                all_outputs = [outputs_orig]

                # Process original images through the PIL pipeline for augmentations
                batch_pil = [transforms.ToPILImage()(img) for img in inputs.cpu()]

                # Apply each TTA transform
                for transform in tta_transforms[1:num_augmentations]:  # Skip the first (already applied)
                    # Apply transform to each image in batch
                    batch_aug = torch.stack([
                        transform(img) for img in batch_pil
                    ]).to(DEVICE)

                    # Get predictions for this augmentation
                    with autocast():
                        outputs_aug = model(batch_aug)

                    all_outputs.append(outputs_aug)

                # Average predictions across all augmentations
                outputs = torch.stack(all_outputs).mean(0)
            else:
                outputs = outputs_orig

            # Get predicted labels
            _, preds = torch.max(outputs, 1)

            # Calculate accuracy
            correct += (preds == targets).sum().item()
            total += batch_size

            # Store predictions and targets for metrics
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_targets

# Standard evaluation function
def evaluate(model, loader):
    """Evaluate the model without test-time augmentation"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Forward pass
            with autocast():
                outputs = model(inputs)

            # Get predicted labels
            _, preds = torch.max(outputs, 1)

            # Calculate accuracy
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # Store predictions and targets for metrics
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, all_preds, all_targets

# Visualization functions
def plot_confusion_matrix(y_true, y_pred, classes, normalize=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    plt.figure(figsize=(12, 10))
    fmt = '.2f' if normalize else 'd'
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.savefig(f"{CHECKPOINT_DIR}/confusion_matrix.png", bbox_inches='tight')
    plt.close()

def plot_training_history(train_accs, val_accs, train_losses=None, val_losses=None):
    """Plot training history"""
    plt.figure(figsize=(18, 6))

    # Plot accuracies
    plt.subplot(1, 2, 1)
    plt.plot(train_accs, label='Training Accuracy', linewidth=2)
    plt.plot(val_accs, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Training & Validation Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Plot generalization gap
    plt.subplot(1, 2, 2)
    gaps = [t - v for t, v in zip(train_accs, val_accs)]
    plt.plot(gaps, label='Train-Val Gap', color='red', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Gap (%)', fontsize=12)
    plt.title('Generalization Gap', fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/training_history.png", bbox_inches='tight')
    plt.close()

# Function to save training checkpoints
def save_checkpoint(epoch, model, optimizer, scheduler, val_acc, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc
    }

    # Save regular checkpoint
    # torch.save(checkpoint, f"{CHECKPOINT_DIR}/checkpoint_epoch{epoch+1}.pth")

    # Save as best if it's the best model
    if is_best:
        torch.save(checkpoint, f"{CHECKPOINT_DIR}/best_model.pth")
        print(f"  New best model saved: {val_acc:.2f}%")

# Main training loop with progressive unfreezing
def train_model():
    best_val_acc = 0
    patience_counter = 0
    train_accs = []
    val_accs = []

    # Progressive unfreezing schedule
    unfreeze_schedule = [
        {'epoch': 0, 'percentage': 0.0},  # Start with all layers frozen
        {'epoch': 5, 'percentage': 0.1},  # Unfreeze 10% of the model after 5 epochs
        {'epoch': 10, 'percentage': 0.2},  # Unfreeze 20% of the model after 10 epochs
        {'epoch': 15, 'percentage': 0.3},  # Unfreeze 30% of the model after 15 epochs
        {'epoch': 20, 'percentage': 0.4},  # Unfreeze 50% of the model after 20 epochs
        {'epoch': 25, 'percentage': 0.5}
    ]

    # Train for NUM_EPOCHS
    for epoch in range(NUM_EPOCHS):
        # Check if we need to unfreeze more layers
        for schedule in unfreeze_schedule:
            if epoch == schedule['epoch']:
                print(f"Epoch {epoch+1}: Unfreezing {schedule['percentage']*100}% of the model")
                model.unfreeze_layers(schedule['percentage'])
                break

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, scaler
        )
        train_accs.append(train_acc)

        # Evaluate on validation set
        val_acc, val_preds, val_targets = evaluate(model, test_loader)
        val_accs.append(val_acc)

        # Print epoch results
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Acc = {train_acc:.2f}%")
        print(f"  Val Acc   = {val_acc:.2f}%")
        print(f"  LR = {optimizer.param_groups[0]['lr']:.6f}")

        # Save model if validation accuracy improves
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, val_acc, is_best)

        # Early stopping check
        if patience_counter >= PATIENCE and epoch > 30:  # Ensure we train for at least 30 epochs
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Visualize training progress periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            plot_training_history(train_accs, val_accs)

    # Plot final training history
    plot_training_history(train_accs, val_accs)

    # Return best validation accuracy
    return best_val_acc

# Function to evaluate the best model
def evaluate_best_model():
    # Load the best model
    checkpoint = torch.load(f"{CHECKPOINT_DIR}/best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1} with validation accuracy {checkpoint['val_acc']:.2f}%")

    # Evaluate with test-time augmentation
    print("Performing final evaluation with test-time augmentation...")
    final_acc, y_pred, y_true = tta_evaluate(model, test_loader, num_augmentations=5)
    print(f"\nFinal Test Accuracy with TTA: {final_acc:.2f}%")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, train_dataset.classes)

    return final_acc, y_pred, y_true

# # Function to predict emotion for a single image
def predict_emotion(model, image_path):
    """Predict emotion for a single image with confidence scores"""
    # Ensure model is in evaluation mode
    model.eval()

    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')

    # Apply face detection (optional)
    # This code assumes you have MTCNN installed, if not, comment out or install with:
    # pip install facenet-pytorch
    try:
        face_detector = MTCNN(image_size=IMAGE_SIZE, margin=20, keep_all=False,
                             min_face_size=40, thresholds=[0.6, 0.7, 0.7],
                             factor=0.707, post_process=True, device=DEVICE)

        # Convert PIL to cv2 format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Detect face
        boxes, _ = face_detector.detect(img)
        if boxes is not None:
            # Get the first face
            box = boxes[0]
            x1, y1, x2, y2 = [int(b) for b in box]

            # Add margin
            h, w = img_cv.shape[:2]
            margin = int(((x2-x1) + (y2-y1)) / 8)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            # Crop face
            img_cv = img_cv[y1:y2, x1:x2]

            # Convert back to PIL
            img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"Face detection failed: {e}")
        print("Using original image without face detection")
        pass

    # Apply validation transform
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    # Get prediction
    with torch.no_grad():
        with autocast():
            output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]

    # Get all emotion probabilities
    classes = train_dataset.classes
    emotion_probs = {classes[i]: prob.item() for i, prob in enumerate(probabilities)}

    # Get top prediction
    top_prob, top_class = torch.max(probabilities, 0)
    predicted_emotion = classes[top_class.item()]
    confidence = top_prob.item() * 100

    # Return results
    return {
        'emotion': predicted_emotion,
        'confidence': confidence,
        'all_probabilities': emotion_probs
    }

# Function to visualize model predictions on sample images
def visualize_predictions(model, dataset, num_samples=8):
    """Visualize model predictions on sample images"""
    # Set model to evaluation mode
    model.eval()

    # Get a batch of images
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels = next(iter(dataloader))

    # Get predictions
    images = images.to(DEVICE)
    with torch.no_grad():
        with autocast():
            outputs = model(images)
        _, preds = torch.max(outputs, 1)

    # Convert images for display
    images = images.cpu()

    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    # Plot each image with prediction
    for i, ax in enumerate(axes):
        # Reverse normalization (approximate)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        # Display image
        ax.imshow(img)

        # Get prediction and true label
        pred_label = dataset.classes[preds[i].item()]
        true_label = dataset.classes[labels[i].item()]

        # Display colored text based on whether prediction is correct
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{CHECKPOINT_DIR}/sample_predictions.png", bbox_inches='tight')
    plt.close()

    return fig

# Run the training process
if __name__ == "__main__":
    print("Starting training process...")
    best_val_acc = train_model()

    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    print("\nEvaluating best model...")
    final_acc, y_pred, y_true = evaluate_best_model()

    # Visualize some sample predictions
    print("\nVisualizing sample predictions...")
    visualize_predictions(model, test_dataset)

    print(f"\nFinal test accuracy: {final_acc:.2f}%")
    print("Complete! All results saved to:", CHECKPOINT_DIR)