# export_pytorch_to_onnx.py
import torch, torch.nn as nn, torch.nn.functional as F
import timm
import numpy as np

# ---- keep the same labels/order as training ----
EMOTIONS = ['anger','contempt','disgust','fear','happiness','neutrality','sadness','surprise']

# ---- must match training config in test.py ----
IMG_SIZE = 224
NUM_CLASSES = 8
DROPOUT = 0.3
MODEL_NAME = 'convnext_base'
PTH_PATH = "best_model.pth"          # <-- put your .pth filename here
ONNX_PATH = "face_emotion_pt.onnx"

class SOTAEmotionModel(nn.Module):
    def __init__(self, model_name='convnext_base', num_classes=8, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, drop_rate=dropout_rate, drop_path_rate=0.2
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            self.feature_dim = self.backbone(dummy).shape[1]

        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(self.feature_dim, self.feature_dim // 16), nn.SiLU(),
            nn.Linear(self.feature_dim // 16, self.feature_dim), nn.Sigmoid()
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim, num_heads=16, dropout=0.1, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate), nn.Linear(self.feature_dim, 1024), nn.BatchNorm1d(1024), nn.SiLU(), nn.Dropout(dropout_rate * 0.7),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, NUM_CLASSES)
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); 
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)                              # (B, C)
        se_w = self.se_block(features.unsqueeze(-1))             # (B, C)
        features_se = features * se_w
        att_in = features_se.unsqueeze(1)                        # (B, 1, C)
        attended, _ = self.attention(att_in, att_in, att_in)     # (B, 1, C)
        attended = attended.squeeze(1)                           # (B, C)
        combined = features + features_se + attended             # (B, C)
        return self.classifier(combined)                         # (B, 8)

def main():
    device = torch.device("cpu")
    model = SOTAEmotionModel(MODEL_NAME, NUM_CLASSES, DROPOUT).to(device).eval()

    ckpt = torch.load(PTH_PATH, map_location=device)
    # support both styles: {'model_state_dict': ...} or plain state_dict
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}

    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        export_params=True, opset_version=17, do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    print(f"✅ Exported to {ONNX_PATH}")

if __name__ == "__main__":
    main()
