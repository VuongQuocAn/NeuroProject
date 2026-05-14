import torch
import torchvision.models as models
from e.BrainTumor.NeuroProject.backend.ai_core.architectures.survival_net import MultimodalBrainTumorModel

def check_layers():
    print("Checking DenseNet121 Layers...")
    model = models.densenet121(weights=None)
    
    # Kiem tra xem denseblock3 co layer 24 khong
    try:
        l3 = model.features.denseblock3.denselayer24.conv2
        print("OK: denseblock3.denselayer24.conv2 exists")
    except Exception as e:
        print(f"ERROR: denseblock3.denselayer24.conv2 NOT found: {e}")

    # Kiem tra xem denseblock4 co layer 16 khong
    try:
        l4 = model.features.denseblock4.denselayer16.conv2
        print("OK: denseblock4.denselayer16.conv2 exists")
    except Exception as e:
        print(f"ERROR: denseblock4.denselayer16.conv2 NOT found: {e}")

if __name__ == "__main__":
    check_layers()
