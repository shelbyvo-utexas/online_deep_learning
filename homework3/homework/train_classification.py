import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric

# Config
DATA_PATH = "datasets/classification"  # adjust path
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_loader = load_data(f"{DATA_PATH}/train", batch_size=BATCH_SIZE, shuffle=True)
val_loader = load_data(f"{DATA_PATH}/val", batch_size=BATCH_SIZE, shuffle=False)

# Model, optimizer, loss
model = Classifier().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
metric = AccuracyMetric()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    metric.reset()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        metric.add(logits.argmax(1), labels)
    train_acc = metric.compute()["accuracy"]

    # Validation
    model.eval()
    metric.reset()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            logits = model(images)
            metric.add(logits.argmax(1), labels)
    val_acc = metric.compute()["accuracy"]
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# Save model
save_model(model)
