import torch
import sys
from ai_core.architectures.survival_net import MultimodalBrainTumorModel, CoxPHLoss, c_index_score

def test_model():
    batch_size = 4
    num_genes = 1000
    feature_dim = 128
    
    # Create model
    model = MultimodalBrainTumorModel(num_genes=num_genes, feature_dim=feature_dim)
    model.eval()
    
    # Fake data
    mri = torch.randn(batch_size, 10, 3, 224, 224)
    wsi = torch.randn(batch_size, 5, 3, 224, 224)
    rna = torch.randn(batch_size, num_genes)
    clinical = torch.randn(batch_size, 18)
    
    has_mri = torch.ones(batch_size)
    has_wsi = torch.ones(batch_size)
    has_rna = torch.ones(batch_size)
    has_clinical = torch.ones(batch_size)
    
    # Indices
    mri_mask = torch.ones(batch_size, 10)
    wsi_mask = torch.ones(batch_size, 5)
    
    with torch.no_grad():
        risk_score, attn_weights = model(
            mri, wsi, rna, clinical, 
            has_mri, has_wsi, has_rna, has_clinical,
            mri_mask, wsi_mask
        )
        
    print(f"Risk score shape: {risk_score.shape}")
    print(f"Attn weights shape: {attn_weights.shape}")
    
    # Test loss
    criterion = CoxPHLoss()
    times = torch.tensor([10.0, 20.0, 30.0, 40.0])
    events = torch.tensor([1.0, 0.0, 1.0, 1.0])
    loss = criterion(risk_score, times, events)
    print(f"Loss: {loss.item()}")
    
    # Test C-Index
    c_index = c_index_score(risk_score, times, events)
    print(f"C-Index: {c_index}")
    
    print("Test passed!")

if __name__ == "__main__":
    test_model()
