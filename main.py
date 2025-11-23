import csv
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("dataset")
SUBMISSION_FILE = "submission_debug.csv"

BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

# ----------------------------
# UNZIP
# ----------------------------
if not DATASET_DIR.exists():
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

# ----------------------------
# VÉRIFICATION DU DATASET
# ----------------------------
print("\n" + "="*60)
print("DATASET VERIFICATION")
print("="*60)

from torchvision import datasets

# Charger sans transform d'abord
temp_dataset = datasets.ImageFolder(DATASET_DIR / "train")
print(f"\nClasses detected: {temp_dataset.classes}")
print(f"Class to idx: {temp_dataset.class_to_idx}")
print(f"Number of images per class:")
for class_name in temp_dataset.classes:
    class_dir = DATASET_DIR / "train" / class_name
    n_images = len(list(class_dir.glob("*")))
    print(f"  {class_name}: {n_images} images")

# Vérifier quelques images
print("\nSample images:")
for i in range(min(5, len(temp_dataset))):
    img_path, label = temp_dataset.imgs[i]
    print(f"  {Path(img_path).name} → class {label} ({temp_dataset.classes[label]})")

# ----------------------------
# TRANSFORMS SIMPLES
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------
# DATASETS
# ----------------------------
train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=eval_transform)

class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.files = sorted(list(self.root.glob("*.*")))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.name

test_dataset = TestDataset(DATASET_DIR / "test", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"\nDataLoaders ready:")
print(f"  Train: {len(train_dataset)} images")
print(f"  Val: {len(val_dataset)} images")
print(f"  Test: {len(test_dataset)} images")

# ----------------------------
# MODÈLE CNN PUR (SANS FOURIER)
# ----------------------------
class PureCNNDetector(nn.Module):
    """CNN pur pour debugging - pas de features Fourier"""
    
    def __init__(self):
        super().__init__()
        
        # Utiliser ResNet18 pré-entraîné (plus stable que MobileNet)
        self.backbone = models.resnet18(weights='DEFAULT')
        
        # Remplacer la dernière couche
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 2)
        
        # Fine-tuning: freeze les premières couches
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

# ----------------------------
# TRAINING FUNCTIONS
# ----------------------------
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for images, labels in pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Vérifier que la loss n'explose pas
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN/Inf detected in loss!")
            continue
        
        loss.backward()
        
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.1f}%'
        })
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    acc = 100. * correct / total
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        idx = np.where(fpr <= 0.01)[0]
        tpr_at_1fpr = tpr[idx[-1]] if len(idx) > 0 else 0.0
    except:
        auc = 0.0
        tpr_at_1fpr = 0.0
    
    # Distribution des prédictions
    pred_dist = {
        '<0.3': np.sum(all_probs < 0.3),
        '0.3-0.7': np.sum((all_probs >= 0.3) & (all_probs < 0.7)),
        '>0.7': np.sum(all_probs >= 0.7)
    }
    
    return acc, auc, tpr_at_1fpr, pred_dist

# ----------------------------
# TRAINING
# ----------------------------
print("\n" + "="*60)
print("TRAINING - PURE CNN")
print("="*60)

model = PureCNNDetector().to(DEVICE)

# Compter les paramètres entraînables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters:")
print(f"  Trainable: {trainable_params:,}")
print(f"  Total: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

best_auc = 0.0
patience = 0
max_patience = 15

history = {
    'train_loss': [],
    'train_acc': [],
    'val_acc': [],
    'val_auc': []
}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
    val_acc, val_auc, tpr, pred_dist = validate(model, val_loader)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)
    
    print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
    print(f"Val: Acc={val_acc:.1f}%, AUC={val_auc:.4f}, TPR@1%FPR={tpr:.4f}")
    print(f"Predictions distribution: {pred_dist}")
    
    scheduler.step(val_auc)
    
    if val_auc > best_auc:
        improvement = val_auc - best_auc
        best_auc = val_auc
        patience = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'auc': val_auc,
            'acc': val_acc
        }, "best_pure_cnn.pth")
        print(f"✓ Saved model (AUC improved by {improvement:.4f})")
    else:
        patience += 1
        print(f"No improvement ({patience}/{max_patience})")
        
        if patience >= max_patience:
            print("\n⚠️ Early stopping!")
            break

print(f"\n{'='*60}")
print(f"Best validation AUC: {best_auc:.4f}")
print(f"{'='*60}")

# Plot training history
if len(history['val_auc']) > 1:
    print("\nTraining history:")
    print("Epoch | Train Loss | Train Acc | Val Acc | Val AUC")
    print("-" * 60)
    for i in range(len(history['val_auc'])):
        print(f"{i+1:5d} | {history['train_loss'][i]:10.4f} | {history['train_acc'][i]:9.1f} | {history['val_acc'][i]:7.1f} | {history['val_auc'][i]:7.4f}")

# ----------------------------
# LOAD BEST MODEL
# ----------------------------
checkpoint = torch.load("best_pure_cnn.pth", weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"\n✓ Loaded best model from epoch {checkpoint['epoch']}")

# Re-validate pour vérifier
final_acc, final_auc, final_tpr, final_dist = validate(model, val_loader)
print(f"\nFinal validation:")
print(f"  Accuracy: {final_acc:.2f}%")
print(f"  AUC: {final_auc:.4f}")
print(f"  TPR@1%FPR: {final_tpr:.4f}")
print(f"  Distribution: {final_dist}")

# ----------------------------
# INFERENCE ON TEST SET
# ----------------------------
print("\n" + "="*60)
print("INFERENCE ON TEST SET")
print("="*60)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for images, names in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        scores = probs[:, 1].cpu().numpy()
        
        image_names.extend(names)
        predictions.extend(scores)

predictions = np.array(predictions)

print(f"\nTest predictions statistics:")
print(f"  Min: {predictions.min():.4f}")
print(f"  Q1: {np.percentile(predictions, 25):.4f}")
print(f"  Median: {np.median(predictions):.4f}")
print(f"  Q3: {np.percentile(predictions, 75):.4f}")
print(f"  Max: {predictions.max():.4f}")
print(f"  Mean: {predictions.mean():.4f}")
print(f"  Std: {predictions.std():.4f}")

# Distribution
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
hist, _ = np.histogram(predictions, bins=bins)
print(f"\nDistribution:")
for i in range(len(bins)-1):
    count = hist[i]
    pct = 100 * count / len(predictions)
    print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}]: {count:4d} images ({pct:5.1f}%)")

# ----------------------------
# SAVE SUBMISSION
# ----------------------------
df = pd.DataFrame({
    "image_name": image_names,
    "score": predictions
})

df = df.sort_values('image_name').reset_index(drop=True)
df.to_csv(SUBMISSION_FILE, index=False)

print(f"\n✓ Saved {SUBMISSION_FILE}")
print("\nFirst 20 predictions:")
print(df.head(20))

print("\n✅ Done!")
