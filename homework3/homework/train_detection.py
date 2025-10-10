import torch
import torch.nn as nn
import torch.nn.functional as F
from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data
from homework.metrics import DetectionMetric

# Config
DATA_PATH = "drive_data"  # adjust path
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader = load_data(f"{DATA_PATH}/train", batch_size=BATCH_SIZE, shuffle=True)
val_loader = load_data(f"{DATA_PATH}/val", batch_size=BATCH_SIZE, shuffle=False)

# Model, optimizer, loss
model = Detector().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion_seg = nn.CrossEntropyLoss()
criterion_depth = nn.MSELoss()
metric = DetectionMetric()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    metric.reset()
    for images, seg_labels, depth_labels in train_loader:
        images = images.to(DEVICE)
        seg_labels = seg_labels.to(DEVICE)
        depth_labels = depth_labels.to(DEVICE)

        optimizer.zero_grad()
        seg_logits, depth_pred = model(images)

        # Upsample outputs to match label sizes
        seg_logits = F.interpolate(seg_logits, size=seg_labels.shape[1:], mode='bilinear', align_corners=False)
        depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=seg_labels.shape[1:], mode='bilinear', align_corners=False).squeeze(1)

        # Compute losses
        seg_loss = criterion_seg(seg_logits, seg_labels)
        depth_loss = criterion_depth(depth_pred, depth_labels)
        loss = seg_loss + depth_loss

        loss.backward()
        optimizer.step()

        metric.add(seg_logits.argmax(1), seg_labels, depth_pred, depth_labels)

    train_metrics = metric.compute()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train metrics: {train_metrics}")

    # Validation
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, seg_labels, depth_labels in val_loader:
            images = images.to(DEVICE)
            seg_labels = seg_labels.to(DEVICE)
            depth_labels = depth_labels.to(DEVICE)

            seg_logits, depth_pred = model(images)
            seg_logits = F.interpolate(seg_logits, size=seg_labels.shape[1:], mode='bilinear', align_corners=False)
            depth_pred = F.interpolate(depth_pred.unsqueeze(1), size=seg_labels.shape[1:], mode='bilinear', align_corners=False).squeeze(1)

            metric.add(seg_logits.argmax(1), seg_labels, depth_pred, depth_labels)

    val_metrics = metric.compute()
    print(f"Epoch {epoch+1}/{EPOCHS} | Val metrics: {val_metrics}")

# Save model
save_model(model)
