#!/usr/bin/env python3
"""
WATERMARK DETECTION - ConvNeXt-Base
Architecture  moderne
"""

import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm

# ================================================================
# CONFIG
# ================================================================
ZIP_FILE = "Dataset.zip"
DATASET_DIR = Path("dataset")
CHECKPOINT_PATH = "convnext_base_watermark.pth"
SUBMISSION_FILE = "submission_convnext.csv"

BATCH_SIZE = 24
NUM_EPOCHS = 6
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔥 Device: {DEVICE}\n")

# ================================================================
# UNZIP
# ================================================================
if not DATASET_DIR.exists():
    print("📦 Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

# ================================================================
# TRANSFORMS
# ================================================================
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================================================================
# DATASETS
# ================================================================
train_dataset = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
val_dataset = datasets.ImageFolder(DATASET_DIR / "val", transform=eval_transform)

class TestDataset(Dataset):
    def __init__(self, root, transform):
        self.files = sorted(list(Path(root).glob("*.*")))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), self.files[idx].name

test_dataset = TestDataset(DATASET_DIR / "test", transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"📊 Dataset:")
print(f"   Train: {len(train_dataset)}")
print(f"   Val:   {len(val_dataset)}")
print(f"   Test:  {len(test_dataset)}\n")

# ================================================================
# MODEL: ConvNeXt-Base
# ================================================================
class ConvNeXtBase(nn.Module):
    """ConvNeXt-Base - Architecture moderne state-of-the-art"""
    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_base(weights='DEFAULT')
        num_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(num_features, 1)
    
    def forward(self, x):
        return self.backbone(x)

# ================================================================
# TRAINING FUNCTIONS
# ================================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            
            all_probs.extend(probs.cpu().numpy() if probs.dim() > 0 else [probs.cpu().item()])
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    # TPR@1%FPR
    idx = np.where(fpr <= 0.01)[0]
    tpr_at_1fpr = tpr[idx[-1]] if len(idx) > 0 else 0.0
    threshold_at_1fpr = thresholds[idx[-1]] if len(idx) > 0 else 0.5
    
    # Accuracy at optimal threshold
    preds_optimal = (all_probs >= threshold_at_1fpr).astype(int)
    acc_optimal = 100. * (preds_optimal == all_labels).mean()
    
    return auc, tpr_at_1fpr, threshold_at_1fpr, acc_optimal

# ================================================================
# TRAINING
# ================================================================
print("="*70)
print("🚀 TRAINING ConvNeXt-Base")
print("="*70)

model = ConvNeXtBase().to(DEVICE)

# Model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Model parameters:")
print(f"   Total:      {total_params:,}")
print(f"   Trainable:  {trainable_params:,}\n")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

best_tpr = 0.0
best_epoch = 0
patience = 0
max_patience = 8

print("Epoch | Loss   | AUC    | TPR@1%FPR | Threshold | Acc@Th | Status")
print("-" * 70)

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    
    # Validate
    val_auc, val_tpr, val_threshold, val_acc = validate(model, val_loader)
    
    # Scheduler
    scheduler.step()
    
    # Save best
    status = ""
    if val_tpr > best_tpr:
        improvement = val_tpr - best_tpr
        best_tpr = val_tpr
        best_epoch = epoch
        patience = 0
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        status = f"✅ BEST (+{improvement:.4f})"
    else:
        patience += 1
        status = f"⏳ No improv ({patience}/{max_patience})"
    
    # Print
    print(f"{epoch:5d} | {train_loss:.4f} | {val_auc:.4f} | "
          f"{val_tpr:9.4f} | {val_threshold:9.4f} | {val_acc:6.2f} | {status}")
    
    # Early stopping
    if patience >= max_patience:
        print(f"\n⚠️ Early stopping at epoch {epoch}")
        break

print("\n" + "="*70)
print(f"🏆 BEST TPR@1%FPR: {best_tpr:.4f} (epoch {best_epoch})")
print("="*70)

# ================================================================
# FINAL VALIDATION
# ================================================================
model.load_state_dict(torch.load(CHECKPOINT_PATH))
final_auc, final_tpr, final_th, final_acc = validate(model, val_loader)

print(f"\n📊 Final validation with best model:")
print(f"   AUC:         {final_auc:.4f}")
print(f"   TPR@1%FPR:   {final_tpr:.4f} ⭐")
print(f"   Threshold:   {final_th:.4f}")
print(f"   Acc@Th:      {final_acc:.2f}%")

# ================================================================
# INFERENCE ON TEST SET
# ================================================================
print("\n" + "="*70)
print("🔮 INFERENCE ON TEST SET")
print("="*70)

model.eval()
predictions = []
image_names = []

with torch.no_grad():
    for images, names in tqdm(test_loader, desc="Predicting"):
        images = images.to(DEVICE)
        outputs = model(images)
        scores = torch.sigmoid(outputs).squeeze()
        
        if scores.dim() == 0:
            scores = [scores.item()]
        else:
            scores = scores.cpu().numpy()
        
        image_names.extend(names)
        predictions.extend(scores if isinstance(scores, list) else scores.tolist())

predictions = np.array(predictions)

# ================================================================
# STATISTICS
# ================================================================
print(f"\n📊 Test predictions:")
print(f"   Min:    {predictions.min():.4f}")
print(f"   Q25:    {np.percentile(predictions, 25):.4f}")
print(f"   Median: {np.median(predictions):.4f}")
print(f"   Mean:   {predictions.mean():.4f}")
print(f"   Q75:    {np.percentile(predictions, 75):.4f}")
print(f"   Max:    {predictions.max():.4f}")
print(f"   Std:    {predictions.std():.4f}")

# Distribution
print(f"\n📈 Distribution:")
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
hist, _ = np.histogram(predictions, bins=bins)
max_count = max(hist)

for i in range(len(bins)-1):
    count = hist[i]
    pct = 100 * count / len(predictions)
    bar_length = int((count / max_count) * 40) if max_count > 0 else 0
    bar = "█" * bar_length
    print(f"   [{bins[i]:.1f}-{bins[i+1]:.1f}]: {count:4d} ({pct:5.1f}%) {bar}")

# ================================================================
# SAVE
# ================================================================
df = pd.DataFrame({
    "image_name": image_names,
    "score": predictions
})

df = df.sort_values('image_name').reset_index(drop=True)
df.to_csv(SUBMISSION_FILE, index=False)

print(f"\n✅ Saved {SUBMISSION_FILE}")
print(f"✅ Saved {CHECKPOINT_PATH}")
print(f"\n🏆 Best TPR@1%FPR: {best_tpr:.4f}")
print("\n" + "="*70)
print("✅ DONE!")
print("="*70)

