import torch
import torch.nn as nn
import torch.nn.functional as F

class SliceAttention(nn.Module):
    def __init__(self, feature_dim = 512, hidden_dim = 128):
        super(SliceAttention, self).__init__()

        # Mạng neural nhỏ để tính điểm quan trọng (Attention Score) cho từng lát cắt
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(), # Hàm kích hoạt phi tuyến tính
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1, bias = False) # Output ra 1 con số (điểm số)
        )

    def forward(self, x, mask=None):
        """
        Input x: [Batch, Slices, Feature_Dim] (VD: [8, 30, 512])
        Input mask: [Batch, Slices] (1=Real, 0=Padding)
        """

        # 1. Tính điểm thô (Raw Scores)
        # [Batch, Slices, 512] -> [Batch, Slices, 1]
        attn_scores = self.attention_net(x)

        # 2. Xử lý Masking
        if mask is not None:
            # Mở rộng mask để khớp dimension: [Batch, Slices] -> [Batch, Slices, 1]
            mask_expanded = mask.unsqueeze(-1)

            # Gán điểm âm vô cùng (-1e9) cho các vị trí Padding (mask=0)
            # Để khi qua Softmax, xác suất của nó sẽ = 0
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e4)

        # 2. Softmax để biến điểm thành xác suất (Tổng các trọng số = 1)
        # Tính trên chiều Slices (dim=1)
        # Softmax -> [Batch, Slices, 1]
        attn_weights = F.softmax(attn_scores, dim = 1)

        # 3. Tính tổng có trọng số (Weighted Sum)
        # Feature = w1*Slice1 + w2*Slice2 + ...
        # [Batch, Slices, 1] * [Batch, Slices, 512] -> [Batch, Slices, 512]
        # Sum theo chiều Slices -> [Batch, 512]
        aggregated_features = torch.sum(attn_weights * x, dim = 1)

        return aggregated_features

class ClinicalEncoder(nn.Module):
    """Mã hóa dữ liệu lâm sàng thành feature vector 512 chiều"""
    def __init__(self, input_dim=18, output_dim=512):
        super(ClinicalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
            nn.ELU()
        )

    def forward(self, x):
        return self.net(x)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class ImageEncoder(nn.Module):
    """
    Lớp mã hóa ảnh sử dụng DenseNet121 làm backbone.
    Mục đích: Trích xuất đặc trưng từ ảnh 2D (MRI hoặc WSI) thành vector 1D.
    Tối ưu cho y tế nhờ khả năng giữ lại đặc trưng không gian ở mọi cấp độ (Dense blocks).
    """

    def __init__(self, output_dim=512, freeze_backbone=True):
        """
        Khởi tạo ImageEncoder.

        Args:
            output_dim (int): Kích thước của vector đặc trưng đầu ra mong muốn.
            freeze_backbone (bool): Nếu True, đóng băng các trọng số của DenseNet để tránh phá vỡ pre-trained weights.
        """
        super(ImageEncoder, self).__init__()

        print("Initializing ImageEncoder with DenseNet121 backbone...")

        # 1. Load Pre-trained DenseNet121
        densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Lấy phần trích xuất đặc trưng (bỏ qua lớp classifier cuối cùng)
        self.feature_extractor = densenet.features

        # # --- BẬT CHECKPOINTING NẾU ĐƯỢC YÊU CẦU ---
        # self.use_checkpointing = use_checkpointing
        # if self.use_checkpointing:
        #     # Gradient checkpointing yêu cầu inputs phải có requires_grad=True
        #     # Nên nếu ta đóng băng toàn bộ backbone, checkpointing sẽ bị lỗi.
        #     # Do đó ta chỉ nên đóng băng một phần, hoặc không đóng băng khi dùng checkpointing.
        #     # Trong code này, tôi giả sử bạn đã UNFREEZE toàn bộ mô hình như trong log.
        #     pass

        # DenseNet121 luôn trả ra số lượng channels là 1024 ở lớp cuối
        in_features = 1024

        # 2. Đóng băng trọng số
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            print("Backbone layers frozen")

        # 3. Các lớp xử lý hậu kỳ (Projection Head)
        # AdaptiveAvgPool2d đưa các feature map (vd 8x8) về kích thước (1x1) một cách linh hoạt
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features, output_dim)
        self.act = nn.ELU()

        # Attention để gom các slice lại
        self.slice_attention = SliceAttention(feature_dim=output_dim)

    def forward(self, x, slice_mask=None):
        """
        Thực hiện tính toán lan truyền xuôi.

        Args:
            x (Tensor): Tensor ảnh có kích thước [Batch, Num_Slices, Channels, Height, Width].
            slice_mask (Tensor, optional): Mặt nạ đánh dấu các lát cắt hợp lệ.

        Returns:
            Tensor: Vector đặc trưng đã được gom nhóm, kích thước [Batch, output_dim].
        """
        b, s, c, h, w = x.shape

        if s == 0:
            return torch.zeros(b, self.fc.out_features, device=x.device)

        # Gộp Batch và Slices để xử lý song song [B*S, C, H, W]
        x = x.view(b * s, c, h, w)

        # # --- ÁP DỤNG CHECKPOINTING KHI FORWARD ---
        # if self.use_checkpointing and x.requires_grad:
        #     # dummy function để truyền vào checkpoint
        #     def custom_forward(inputs):
        #         return self.feature_extractor(inputs)

        #     # PyTorch cảnh báo về việc dùng use_reentrant=False trong các bản mới
        #     features = checkpoint.checkpoint(custom_forward, x, use_reentrant=False)
        # else:
        #     features = self.feature_extractor(x)

        # Trích xuất đặc trưng qua DenseNet
        features = self.feature_extractor(x)

        # DenseNet yêu cầu chạy ReLU sau khối features cuối cùng
        features = F.relu(features, inplace=False)

        # Global Average Pooling để làm phẳng [B*S, 1024, 1, 1]
        features = self.global_pool(features)
        features = features.view(features.size(0), -1) # Flatten -> [B*S, 1024]

        # Đưa qua lớp Linear [B*S, 512]
        features = self.dropout(features)
        features = self.fc(features)
        features = self.act(features)

        # Tách lại Batch và Slices [Batch, Slices, 512]
        features = features.view(b, s, -1)

        # Attention Pooling gom Slices lại thành 1 vector duy nhất cho bệnh nhân
        out = self.slice_attention(features, mask=slice_mask)

        return out

class OmicEncoder(nn.Module):
    """
    Encodes RNA-seq expression data (vector) into a fixed-size feature vector.
    Uses a Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout.
    """

    def __init__(self, num_genes, output_dim = 512, hidden_dim = 1024):
        """
        Args:
            num_genes (int): Input dimension (number of genes in the dataset).
            output_dim (int): Output feature size (must match ImageEncoder output).
            hidden_dim (int): Size of the intermediate hidden layer.
        """

        super(OmicEncoder, self).__init__()

        print(f"Initializing OmicEncoder: Input {num_genes} -> Hidden {hidden_dim} -> Output {output_dim}")

        # Layer 1: Nén dữ liệu bước đầu (High Dim -> Hidden Dim)
        # Sử dụng Batch Normalization để ổn định dữ liệu gen (vốn có dải giá trị rất rộng)
        self.layer1 = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(p = 0.6)
        )

        # Layer 2: Nén xuống vector đặc trưng (Hidden Dim -> Output Dim)
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
            nn.Dropout(p = 0.4) # Dropout cao (0.4) vì số lượng mẫu ít (tránh học vẹt)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming Normal giúp trọng số phù hợp với hàm activation ELU/ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Gene expression vector [Batch, num_genes]
        Returns:
            Tensor: Feature vector [Batch, output_dim]
        """

        x = self.layer1(x)
        x = self.layer2(x)

        return x

class AttentionFusion(nn.Module):
    """
    Multimodal Gated Attention Fusion.
    Learns to weigh the importance of each modality dynamically.
    Handles missing data by masking attention scores.
    """

    def __init__(self, feature_dim = 512, hidden_dim = 128):
        """
        Args:
            feature_dim (int): Input dimension of feature vectors (must match Encoder output).
            hidden_dim (int): Hidden dimension for the attention mechanism.
        """

        super(AttentionFusion, self).__init__()

        print(f"Initializing AttentionFusion: Input {feature_dim} -> Output {feature_dim}")

        # Mạng Neural nhỏ để tính điểm Attention (Attention Score)
        # Input: Feature Vector -> Output: Scalar Score (1 con số)
        self.attn_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias = False) # Bias=False để tránh lệch điểm khi input=0
        )

    def forward(self, feature_stack, mask = None):
        """
        Args:
            features_stack (Tensor): Stacked features [Batch, Num_Modalities, Feature_Dim]
                                     (Ví dụ: [16, 3, 512])
            mask (Tensor): Boolean mask indicating available modalities [Batch, Num_Modalities]
                           (1 = Available, 0 = Missing)
        Returns:
            fused_feature (Tensor): Weighted sum [Batch, Feature_Dim]
            attention_weights (Tensor): Weights used [Batch, Num_Modalities] (để vẽ XAI)
        """

        # 1. Tính điểm thô (Raw Scores) cho từng nhánh
        # [Batch, 3, 512] -> [Batch, 3, 1]
        attn_scores = self.attn_net(feature_stack)

        # [Batch, 3, 1] -> [Batch, 3]
        attn_scores = attn_scores.squeeze(-1)

        # 2. Xử lý dữ liệu thiếu (Masking)
        if mask is not None:
            # Với những nhánh bị thiếu (mask=0), ta gán điểm attention = âm vô cùng (-1e9)
            # Để khi qua Softmax, trọng số của nó sẽ bằng 0 tuyệt đối.
            # Lưu ý: mask đang là 1/0, ta cần biến đổi chút

            # Tạo mask cho các vị trí bằng 0
            extended_mask = (mask == 0)

            # Fill giá trị -1e9 vào những chỗ bị thiếu
            attn_scores = attn_scores.masked_fill(extended_mask, -1e4)

        # 3. Tính trọng số (Softmax)
        # Biến đổi điểm thành xác suất (tổng = 1)
        # dim=1 là chiều Modalities
        attn_weights = F.softmax(attn_scores, dim = 1)

        # [Batch, 3] -> [Batch, 3, 1] để nhân broadcasting
        attn_weights_expanded = attn_weights.unsqueeze(-1)

        # 4. Tổng hợp (Weighted Sum)
        # Feature tổng = w1*MRI + w2*WSI + w3*RNA
        # [Batch, 3, 1] * [Batch, 3, 512] -> Sum theo chiều Modality -> [Batch, 512]
        fused_feature = torch.sum(attn_weights_expanded * feature_stack, dim = 1)

        return fused_feature, attn_weights

class MultimodalBrainTumorModel(nn.Module):
    """
    The main model architecture that combines MRI, WSI, and RNA-seq.
    """

    def __init__(self, num_genes, feature_dim = 512):
        """
        Args:
            num_genes (int): Input dimension for RNA data.
            feature_dim (int): Common feature vector size (default 512).
        """

        super(MultimodalBrainTumorModel, self).__init__()

        # 1. Các nhánh Encoders
        # MRI Encoder
        self.mri_encoder = ImageEncoder(output_dim = feature_dim)

        # WSI Encoder
        self.wsi_encoder = ImageEncoder(output_dim = feature_dim)

        # RNA Encoder
        if num_genes is not None:
            self.rna_encoder = OmicEncoder(num_genes=num_genes, output_dim=feature_dim)
        else:
            self.rna_encoder = None

        # Thêm Clinical Encoder (18 input -> 512 output)
        self.clinical_encoder = ClinicalEncoder(input_dim=18, output_dim=feature_dim)


        # 2. Fusion Module
        self.fusion = AttentionFusion(feature_dim = feature_dim)

        # 3. Prediction Head (Đầu ra)
        # Input: Fused Feature (512) -> Output: Risk Score (1)
        # Dùng một mạng nhỏ 2 lớp để dự đoán tốt hơn
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Đầu ra CoxPH Risk Score
        )

    def forward(self, mri, wsi, rna, clinical, has_mri, has_wsi, has_rna, has_clinical, mri_mask=None, wsi_mask=None):
        """
        Args:
            Các input tensor từ DataLoader.
            has_...: Tensor [Batch] chứa 0 hoặc 1.
        """

        # if mri.requires_grad == False and self.training:
        #     mri.requires_grad_(True)

        # if wsi.requires_grad == False and self.training:
        #     wsi.requires_grad_(True)

        # --- BƯỚC 1: ENCODE ---
        # những mẫu input=0 (đen xì) thì feature ra cũng vô nghĩa, sẽ bị Mask chặn lại sau
        feat_mri = self.mri_encoder(mri, slice_mask = mri_mask)
        feat_wsi = self.wsi_encoder(wsi, slice_mask = wsi_mask)

        if self.rna_encoder is not None:
            feat_rna = self.rna_encoder(rna)
        else:
            # Define 'b' (batch size) if 'rna_encoder' is None
            # This assumes 'mri' is always present and has a batch dimension
            if mri.dim() > 0: # Check if mri has a batch dimension
                b = mri.shape[0]
            else: # If mri is empty or a scalar, assume batch size 1 or 0 for consistency
                b = 0 # Or handle appropriately based on context
            feat_rna = torch.zeros(b, feat_mri.size(-1), device=rna.device)

        # Chạy Clinical Encoder
        feat_clinical = self.clinical_encoder(clinical)

        # --- BƯỚC 2: STACK & MASK ---
        # Xếp chồng 3 vector lại: [Batch, 4, 512]
        features_stack = torch.stack([feat_mri, feat_wsi, feat_rna, feat_clinical], dim=1)

        # Xếp chồng mask lại: [Batch, 4]
        modality_mask = torch.stack([has_mri, has_wsi, has_rna, has_clinical], dim=1)

        # --- BƯỚC 3: FUSION ---
        # fused_feat: [Batch, 512]
        # attn_weights: [Batch, 3] (Để vẽ biểu đồ xem nó tin vào cái nào)

        fused_feature, attn_weights = self.fusion(features_stack, mask=modality_mask)
        # --- BƯỚC 4: PREDICT ---
        # output: [Batch, 1]
        risk_score = self.classifier(fused_feature)

        return risk_score.squeeze(-1), attn_weights


from IPython.core.magics.namespace import re
class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Loss (Negative Log Likelihood).
    Handle tied times and censored data effectively.
    """

    def __init__(self):
        super(CoxPHLoss, self).__init__()

    def forward(self, risk_scores, times, events):
        """
        Args:
            risk_scores (Tensor): Predicted risk [Batch, 1]
            times (Tensor): Survival time [Batch]
            events (Tensor): Event indicator (1=Dead, 0=Alive/Censored) [Batch]
        """

        # Sắp xếp dữ liệu theo thời gian giảm dần (Yêu cầu bắt buộc của Cox Loss)
        # sort_idx: index đã sắp xếp
        sorted_times, sort_idx = torch.sort(times, descending = True)
        sorted_risk = risk_scores[sort_idx]
        sorted_events = events[sort_idx]

        # Chặn giá trị risk không quá lớn để tránh exp() ra Inf
        # log(float32.max) ~ 88. Chặn ở 20 là rất an toàn cho y tế.
        sorted_risk = torch.clamp(sorted_risk, min=-20, max=20)


        # Tính LogSumExp (Phần mẫu số trong công thức Cox)
        # R(t_i) là tập hợp những người còn sống tại thời điểm t_i
        # exp_risk: e^(h_i)
        exp_risk = torch.exp(sorted_risk)


        # Cumsum ngược (từ dưới lên) để tính tổng các e^h của những người sống lâu hơn
        # cumsum_risk[i] = tổng exp_risk của tất cả bệnh nhân j mà time[j] <= time[i] (sai logic gốc)
        # Logic đúng: Risk Set là những người có time >= time[i].
        # Vì ta đã sort giảm dần time, nên Risk Set của i chính là các phần tử từ 0 đến i (nếu sort tăng dần).
        # Nhưng ở đây ta sort GIẢM DẦN (time lớn đứng trước).
        # Risk Set tại thời điểm t (nhỏ) bao gồm tất cả những người sống lâu hơn t (time lớn hơn).
        # Vậy Risk Set của phần tử i (time nhỏ) bao gồm các phần tử j từ 0 đến i.
        risk_set_sum = torch.cumsum(exp_risk, dim = 0)

        # Lấy log của Risk Set
        # Thêm epsilon nhỏ xíu để log không bị lỗi log(0)
        log_risk_set = torch.log(risk_set_sum + 1e-8)

        # Công thức: Loss = - sum(risk_i - log(sum(risk_set))) chỉ tính trên những người có event=1
        # Lưu ý: sorted_risk chính là log-risk (h_i)

        # Chỉ tính loss cho những người đã chết (Uncensored)
        uncensored_loss = (sorted_risk - log_risk_set) * sorted_events

        # Tổng loss chia cho số lượng người chết (để normalize)
        num_events = torch.sum(sorted_events)

        if num_events == 0:
            return torch.tensor(0.0, device = risk_scores.device, requires_grad = True)

        loss = -torch.sum(uncensored_loss) / num_events

        return loss

# !pip install lifelines
from lifelines.utils import concordance_index

def c_index_score(risk_scores, times, events):
    """
    Tính C-Index dùng thư viện chuẩn Lifelines.
    """
    try:
        # Chuyển sang numpy nếu là tensor
        if isinstance(risk_scores, torch.Tensor):
            risk_scores = risk_scores.detach().cpu().numpy()
        if isinstance(times, torch.Tensor):
            times = times.detach().cpu().numpy()
        if isinstance(events, torch.Tensor):
            events = events.detach().cpu().numpy()

        # Lifelines document: concordance_index(event_times, predicted_scores, event_observed=None)
        # predicted_scores: Higher scores should indicate LONGER survival.
        # Cox Model output: Higher scores indicate HIGHER RISK (SHORTER survival).
        # => Do đó ta phải dùng dấu TRỪ (-risk_scores) để đảo ngược lại.
        return concordance_index(times, -risk_scores, events)

    except Exception as e:
        print(f"Lỗi tính C-Index: {e}")
        return 0.5