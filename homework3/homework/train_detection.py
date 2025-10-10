import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.datasets import road_dataset  # your dataset module

# ------------------------------
# Detector model
# ------------------------------
class Detector(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Simple example architecture, modify as needed
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)  # downsample by 2
        )
        self.seg_head = nn.Conv2d(32, num_classes, 1)  # segmentation logits
        self.depth_head = nn.Conv2d(32, 1, 1)          # depth regression

    def forward(self, x):
        feat = self.encoder(x)
        seg_logits = self.seg_head(feat)      # (B, C, H, W)
        depth_preds = self.depth_head(feat)   # (B, 1, H, W)
        return seg_logits, depth_preds

    @torch.inference_mode()
    def predict(self, x: torch.Tensor):
        """ Required by grader. Returns (seg_labels, depth_preds) """
        self.eval()
        seg_logits, depth_preds = self.forward(x)

        # Upsample to match input resolution
        H, W = x.shape[2], x.shape[3]
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)
        depth_preds = F.interpolate(depth_preds, size=(H, W), mode="bilinear", align_corners=False)

        pred = seg_logits.argmax(dim=1)  # (B, H, W)
        if depth_preds.dim() == 4 and depth_preds.shape[1] == 1:
            depth_preds = depth_preds.squeeze(1)  # (B, H, W)
        return pred, depth_preds

# ------------------------------
# Training loop
# ------------------------------
def train_detector(
    model, dataset_path="drive_data/train", batch_size=16, epochs=10, lr=1e-3, device=None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data = road_dataset.load_data(
        dataset_path, num_workers=2, batch_size=batch_size, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    seg_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in data:
            images = batch["image"].to(device)
            seg_labels = batch["track"].to(device)
            depth_labels = batch["depth"].to(device)

            optimizer.zero_grad()
            seg_logits, depth_preds = model(images)

            # Upsample to match label resolution if necessary
            if seg_logits.shape[2:] != seg_labels.shape[1:]:
                seg_logits = F.interpolate(seg_logits, size=seg_labels.shape[1:], mode="bilinear", align_corners=False)
            if depth_preds.shape[2:] != depth_labels.shape[1:]:
                depth_preds = F.interpolate(depth_preds, size=depth_labels.shape[1:], mode="bilinear", align_corners=False)

            # Segmentation loss
            seg_loss = seg_criterion(seg_logits, seg_labels.long())
            # Depth loss
            if depth_preds.dim() == 4 and depth_preds.shape[1] == 1:
                depth_preds = depth_preds.squeeze(1)
            depth_loss = depth_criterion(depth_preds, depth_labels.float())

            loss = seg_loss + depth_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")

# ------------------------------
# Model loader for grader
# ------------------------------
def load_model(kind: str, with_weights=True):
    if kind == "detector":
        model = Detector()
        if with_weights:
            # load pretrained weights here if available
            pass
        return model
    raise ValueError(f"Unknown model kind: {kind}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    model = Detector()
    train_detector(model)
