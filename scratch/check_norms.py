import torch
import torch.nn as nn
import torch.nn.functional as F

class SliceAttention(nn.Module):
    def __init__(self, feature_dim = 512, hidden_dim = 128):
        super(SliceAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias = False)
        )
    def forward(self, x, mask=None):
        attn_scores = self.attention_net(x)
        attn_weights = F.softmax(attn_scores, dim = 1)
        return torch.sum(attn_weights * x, dim = 1)

class OmicEncoder(nn.Module):
    def __init__(self, num_genes, output_dim = 512, hidden_dim = 1024):
        super(OmicEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(p = 0.6)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
            nn.Dropout(p = 0.4)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super(ImageEncoder, self).__init__()
        self.fc = nn.Linear(1024, output_dim)
        self.act = nn.ELU()
        self.slice_attention = SliceAttention(feature_dim=output_dim)
    def forward(self, x):
        # Mock features from DenseNet [Batch, Slices, 1024]
        b, s, c = x.shape
        x = x.view(b*s, -1)
        x = self.fc(x)
        x = self.act(x)
        x = x.view(b, s, -1)
        return self.slice_attention(x)

def check_norms():
    num_genes = 60664
    rna_enc = OmicEncoder(num_genes)
    img_enc = ImageEncoder()
    
    # Random inputs
    rna_in = torch.randn(1, num_genes)
    # Mock DenseNet output (after global pool) [1, 10, 1024]
    img_in = torch.randn(1, 10, 1024)
    
    with torch.no_grad():
        feat_rna = rna_enc(rna_in)
        feat_img = img_enc(img_in)
        
        print(f"RNA norm: {torch.norm(feat_rna).item():.4f}")
        print(f"IMG norm: {torch.norm(feat_img).item():.4f}")
        
        # Attention score network
        attn_net = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1, bias = False)
        )
        
        # Calculate scores
        score_rna = attn_net(feat_rna)
        score_img = attn_net(feat_img)
        
        print(f"RNA Score: {score_rna.item():.4f}")
        print(f"IMG Score: {score_img.item():.4f}")
        
        weights = F.softmax(torch.tensor([score_img.item(), score_rna.item()]), dim=0)
        print(f"Weights (Img, RNA): {weights.tolist()}")

if __name__ == "__main__":
    check_norms()
