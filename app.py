import os
import sys
# Prepend local sub-project folder to path to avoid resolving conflicts with parent directory modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

try:
    from models.vit_classifier import ViTClassifier
    from models.segformer import SegFormerSegmentation
    from models.moe.moe_layer import MoELayer
    from training.train_moe import replace_segformer_ffn_with_moe, get_moe_auxiliary_loss
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.vit_classifier import ViTClassifier
    from models.segformer import SegFormerSegmentation
    from models.moe.moe_layer import MoELayer
    from training.train_moe import replace_segformer_ffn_with_moe, get_moe_auxiliary_loss

# Set page config
st.set_page_config(
    page_title="ViT & MoE Research Framework",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
        color: #f1f3f9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #161e2f;
        border-radius: 4px;
        color: #a0aec0;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        font-size: 15px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3182ce;
        color: #ffffff !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #63b3ed;
    }
    .header-style {
        font-size: 32px;
        font-weight: 700;
        color: #63b3ed;
        margin-bottom: 20px;
        border-bottom: 2px solid #2d3748;
        padding-bottom: 10px;
    }
    .card {
        background-color: #161e2f;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #2d3748;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading to prevent slow reload times
@st.cache_resource
def load_models_for_dashboard():
    # Instantiate models with pretrained=False to make it fast and completely offline
    vit_model = ViTClassifier(model_name="google/vit-base-patch16-224", num_classes=10, pretrained=False)
    
    segformer_model = SegFormerSegmentation(model_name="nvidia/mit-b0", num_classes=2, pretrained=False)
    
    moe_segformer_model = SegFormerSegmentation(model_name="nvidia/mit-b0", num_classes=2, pretrained=False)
    moe_config = {
        "num_experts": 4,
        "top_k": 2,
        "capacity_factor": 1.2,
        "noisy_gating": True,
        "balance_loss_coef": 0.01
    }
    # Dynamic replacement inside model
    import logging
    dummy_logger = logging.getLogger("dashboard")
    replace_segformer_ffn_with_moe(moe_segformer_model, moe_config, dummy_logger)
    
    # Load trained checkpoints if available
    checkpoint_path = "outputs/medical_segmentation/checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            # Remove module prefix if loaded from distributed trainer
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    cleaned_state_dict[k[7:]] = v
                else:
                    cleaned_state_dict[k] = v
            moe_segformer_model.load_state_dict(cleaned_state_dict)
            st.sidebar.success("Loaded trained MoE model weights!")
        except Exception as e:
            st.sidebar.error(f"Error loading trained MoE weights: {e}")
    else:
        st.sidebar.info("Using randomly initialized MoE model (no trained checkpoint found).")
        
    return vit_model, segformer_model, moe_segformer_model

# Load models
with st.spinner("Initializing models..."):
    vit_model, segformer_model, moe_segformer_model = load_models_for_dashboard()

# Sidebar
st.sidebar.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
st.sidebar.markdown("# **Transformer MoE Research**")
st.sidebar.markdown("---")
st.sidebar.markdown("This interactive panel demonstrates the architectural progression of the graduate Vision Transformer research framework.")
st.sidebar.markdown("### **Framework Phases**")
st.sidebar.markdown("1. **Phase 1:** ViT for Image Classification")
st.sidebar.markdown("2. **Phase 2:** SegFormer for Semantic Segmentation")
st.sidebar.markdown("3. **Phase 3:** SegFormer + Mixture of Experts (MoE)")
st.sidebar.markdown("---")
st.sidebar.markdown("Created by: **Toqeer Ahmed**")

# Main Page Title
st.markdown("<h1 class='header-style'>🔮 Vision Transformer & Mixture of Experts (MoE) Dashboard</h1>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Architectural Comparison", "🎮 Interactive Inference Sandbox", "💻 Implementation Tour"])

with tab1:
    st.markdown("### **Model Size & Parameters Count**")
    st.markdown("Below is the live parameter verification count extracted dynamically from the loaded model architectures:")
    
    # Calculate parameter counts
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    p1_params = count_parameters(vit_model)
    p2_params = count_parameters(segformer_model)
    p3_params = count_parameters(moe_segformer_model)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric(label="Phase 1: ViT Classifier", value=f"{p1_params:,} parameters")
        st.markdown("**Backbone:** ViT-Base-Patch16-224<br>**Input:** 224x224x3 Image<br>**Output:** Classification logits", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric(label="Phase 2: SegFormer", value=f"{p2_params:,} parameters")
        st.markdown("**Backbone:** mit-b0 Segformer<br>**Input:** 224x224x3 Image<br>**Output:** 224x224x2 Segmentation mask", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.metric(label="Phase 3: SegFormer + MoE", value=f"{p3_params:,} parameters")
        st.metric(label="Experts Per MoELayer", value="4 Experts")
        st.markdown("**Backbone:** mit-b0 + 4 experts<br>**Input:** 224x224x3 Image<br>**Output:** MoE-augmented prediction mask", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("### **Structural Differences Table**")
    comparison_data = {
        "Feature": ["Primary Task", "Pretrained Base", "Dynamic Components", "Auxiliary Objectives", "Spatial Context Handling"],
        "Phase 1: ViT Classifier": ["Image Classification", "google/vit-base-patch16-224", "None (Static MLP Heads)", "Cross Entropy Loss Only", "Tokenized patches, position embeddings"],
        "Phase 2: SegFormer": ["Semantic Segmentation", "nvidia/mit-b0", "All-MLP Decoder head", "Cross Entropy Loss Only", "Hierarchical stages, Depthwise Conv (MixFFN)"],
        "Phase 3: SegFormer + MoE": ["MoE-augmented Semantic Segmentation", "nvidia/mit-b0", "MLPExpert Module, GatingNetwork, Top-K Router", "Cross Entropy + Load Balancing Loss", "Gated routing per token, dynamic expert activation"]
    }
    df = pd.DataFrame(comparison_data)
    st.table(df)

    st.markdown("### **MoE Block Diagram**")
    st.markdown("This Mermaid flowchart illustrates the routing execution flow of a single **MoE Layer** substituting the standard MixFFN:")
    st.code("""
    [Input Tokens (N, hidden_dim)] --> [Gating Network] --> [Top-K Router (Logits)]
                                              |
                                              +--> [Calculate Load Balance Penalty]
                                              |
                                     (Select Top-K Experts)
                                              |
      +----------------------+----------------+----------------------+
      |                      |                                       |
    [Expert 1 (MLP)]   [Expert 2 (MLP)]                      [Expert 4 (MLP)]
      |                      |                                       |
      +----------------------+----------------+----------------------+
                                              |
                                     (Gather & Weight gating)
                                              |
                                  [Summed Outputs (N, hidden_dim)]
    """, language="text")

    # Dynamic presentation of training plots if they exist
    st.markdown("---")
    st.markdown("### **📈 Training Performance & Validation Metrics**")
    
    plot_dir = "outputs/medical_segmentation/plots"
    if os.path.exists(plot_dir):
        col_plot1, col_plot2 = st.columns(2)
        loss_curve_path = os.path.join(plot_dir, "loss_curves.png")
        metrics_curve_path = os.path.join(plot_dir, "segmentation_metrics.png")
        
        with col_plot1:
            if os.path.exists(loss_curve_path):
                st.image(loss_curve_path, caption="Training & Validation Loss Curves", use_column_width=True)
            else:
                st.info("Loss curves plot not found.")
        with col_plot2:
            if os.path.exists(metrics_curve_path):
                st.image(metrics_curve_path, caption="Mean IoU & Dice Validation Metrics", use_column_width=True)
            else:
                st.info("Validation metrics plot not found.")
    else:
        st.info("Training plots not found locally. Run training or copy the plots to outputs/medical_segmentation/plots/ to see live training performance charts here.")

    # Dynamic presentation of validation predictions over epochs
    st.markdown("---")
    st.markdown("### **📸 Validation Prediction Epoch Progress**")
    st.markdown("Observe how the MoE-SegFormer model improves its segmentation boundaries over training epochs (Left: Input Image, Center: Ground Truth Mask, Right: Predicted Mask):")
    
    if os.path.exists(plot_dir):
        import glob
        val_preds = glob.glob(os.path.join(plot_dir, "val_predictions_epoch_*.png"))
        if val_preds:
            # Sort files numerically by extracting epoch number
            def extract_epoch(filename):
                try:
                    parts = os.path.basename(filename).split('_')
                    return int(parts[-1].split('.')[0])
                except Exception:
                    return 0
            val_preds_sorted = sorted(val_preds, key=extract_epoch)
            
            # Create a dropdown selector for the epoch prediction visualization
            epoch_labels = [f"Epoch {extract_epoch(p)} Visual Results" for p in val_preds_sorted]
            selected_label = st.selectbox("Select Epoch to View Predictions:", epoch_labels)
            selected_idx = epoch_labels.index(selected_label)
            selected_plot = val_preds_sorted[selected_idx]
            
            st.image(selected_plot, caption=f"Validation Predictions for {selected_label}", use_column_width=True)
        else:
            st.info("No epoch-by-epoch predictions plots found in plots directory.")
    else:
        st.info("Copy the validation predictions plots to outputs/medical_segmentation/plots/ to view epoch progression here.")

with tab2:
    st.markdown("### **Interactive Model Sandbox**")
    st.markdown("Select a research model and generate synthetic datasets to run inference, calculate metrics, and analyze MoE routing distributions.")
    
    task_selected = st.selectbox(
        "Choose Model Task for Inference:",
        ["Phase 1: ViT Classifier (Image Classification)", "Phase 2: SegFormer (Semantic Segmentation)", "Phase 3: SegFormer + MoE (Gated Segmentation)"]
    )
    
    # Simple image generators to create clean shapes
    def generate_synthetic_image(task):
        img_np = np.zeros((224, 224, 3), dtype=np.uint8)
        mask_np = np.zeros((224, 224), dtype=np.uint8)
        
        # Color background
        img_np[:, :, 0] = 50 # Dark Gray Red-tint
        img_np[:, :, 1] = 60
        img_np[:, :, 2] = 80
        
        # Draw a synthetic target object (circle/ellipse)
        rr, cc = np.ogrid[:224, :224]
        center_x, center_y = 112, 112
        radius = 50
        mask = (rr - center_x)**2 + (cc - center_y)**2 < radius**2
        
        # Draw object on image (Greenish pet-shape)
        img_np[mask, 0] = 120
        img_np[mask, 1] = 180
        img_np[mask, 2] = 120
        mask_np[mask] = 1 # Foreground object is class 1
        
        return img_np, mask_np
        
    img_np, mask_np = generate_synthetic_image(task_selected)
    
    st.markdown("#### **Generated Input Data Preview**")
    col_in_img, col_in_mask = st.columns(2)
    with col_in_img:
        st.image(img_np, caption="Synthetic Input Image (RGB)", use_column_width=True)
    with col_in_mask:
        if "Classification" in task_selected:
            st.info("Input label for classification: Class 3 (Synthetic Object)")
        else:
            st.image(mask_np * 255, caption="Ground Truth Target Mask", use_column_width=True)
            
    if st.button("🚀 Run Forward Pass Inference"):
        # Setup inputs
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        # Normalize
        if "Classification" in task_selected:
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
        else:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
        st.markdown("### **Inference Outputs**")
        
        if "Classification" in task_selected:
            vit_model.eval()
            with torch.no_grad():
                logits = vit_model(img_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric(label="Predicted Class", value=f"Class {pred_idx}")
            with c2:
                st.metric(label="Prediction Confidence", value=f"{confidence*100:.2f}%")
                
            # Plot class probabilities
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(10), probs.numpy(), color='#63b3ed')
            ax.set_title("Probability Distribution")
            ax.set_xlabel("Classes")
            ax.set_ylabel("Probability")
            ax.set_facecolor('#161e2f')
            fig.patch.set_facecolor('#0b0f19')
            ax.spines['bottom'].set_color('#2d3748')
            ax.spines['left'].set_color('#2d3748')
            ax.xaxis.label.set_color('#a0aec0')
            ax.yaxis.label.set_color('#a0aec0')
            ax.tick_params(colors='#a0aec0')
            ax.title.set_color('#f1f3f9')
            st.pyplot(fig)
            
        elif "Phase 2" in task_selected:
            segformer_model.eval()
            with torch.no_grad():
                logits = segformer_model(img_tensor)
                pred_mask = logits.argmax(dim=1).squeeze(0).numpy()
                
            c_out1, c_out2 = st.columns(2)
            with c_out1:
                st.image(img_np, caption="Input Image", use_column_width=True)
            with c_out2:
                st.image(pred_mask * 255, caption="Predicted Segmentation Mask", use_column_width=True)
                
        elif "Phase 3" in task_selected:
            moe_segformer_model.eval()
            with torch.no_grad():
                logits = moe_segformer_model(img_tensor)
                pred_mask = logits.argmax(dim=1).squeeze(0).numpy()
                
            c_out1, c_out2 = st.columns(2)
            with c_out1:
                st.image(img_np, caption="Input Image", use_column_width=True)
            with c_out2:
                st.image(pred_mask * 255, caption="Predicted MoE Mask", use_column_width=True)
                
            # Collect and visualize gating routing statistics from MoELayers!
            st.markdown("### **🔍 Mixture of Experts Routing Statistics**")
            st.markdown("When the image was passed through the MoE Segformer, tokens in the encoder stages were dispatched dynamically to different experts. Here is the aggregate token selection distribution across the **8** custom `MoELayer` blocks:")
            
            # Extract routing decisions
            routing_counts = {f"Expert {i+1}": 0 for i in range(4)}
            layer_stats = []
            
            moe_layers = [m for m in moe_segformer_model.modules() if isinstance(m, MoELayer)]
            
            for idx, moe_layer in enumerate(moe_layers):
                if hasattr(moe_layer, 'last_routing_indices'):
                    indices = moe_layer.last_routing_indices # shape: (num_tokens, top_k)
                    # Flatten indices to count selection frequencies
                    flattened_indices = indices.view(-1).numpy()
                    unique, counts = np.unique(flattened_indices, return_counts=True)
                    
                    layer_dict = {f"Expert {i+1}": 0 for i in range(4)}
                    for u, c in zip(unique, counts):
                        layer_dict[f"Expert {u+1}"] = int(c)
                        routing_counts[f"Expert {u+1}"] += int(c)
                    
                    layer_stats.append({
                        "Layer": f"MoELayer {idx+1}",
                        **layer_dict
                    })
            
            if len(layer_stats) > 0:
                # Plot Routing Frequency
                fig_moe, ax_moe = plt.subplots(figsize=(8, 4))
                ax_moe.bar(routing_counts.keys(), routing_counts.values(), color=['#4299e1', '#48bb78', '#ed8936', '#9f7aea'])
                ax_moe.set_title("Total Tokens Dispatched Per Expert (Aggregate)")
                ax_moe.set_ylabel("Token Frequency")
                ax_moe.set_facecolor('#161e2f')
                fig_moe.patch.set_facecolor('#0b0f19')
                ax_moe.spines['bottom'].set_color('#2d3748')
                ax_moe.spines['left'].set_color('#2d3748')
                ax_moe.yaxis.label.set_color('#a0aec0')
                ax_moe.tick_params(colors='#a0aec0')
                ax_moe.title.set_color('#f1f3f9')
                st.pyplot(fig_moe)
                
                # Show Layer-by-layer distribution dataframe
                st.markdown("**Detailed Token Distribution Per Layer:**")
                st.dataframe(pd.DataFrame(layer_stats))
            else:
                st.info("No routing stats available. Make sure model forward pass completed.")

with tab3:
    st.markdown("### **Implementation Details**")
    st.markdown("Inspect the core Python modules executing inside this research framework:")
    
    st.markdown("#### **1. Dynamic FFN-to-MoE Replacement Code**")
    st.markdown("This function traverses the Segformer encoder stages at runtime to substitute standard Feed Forward networks with our shape-compatible `MoELayer` blocks:")
    st.code("""
def replace_segformer_ffn_with_moe(model: SegFormerSegmentation, moe_config: dict, logger) -> int:
    segformer = model.model.segformer
    replaced_count = 0
    
    for stage_idx, stage in enumerate(segformer.stages):
        hidden_dim = segformer.config.hidden_sizes[stage_idx]
        
        for block_idx, block in enumerate(stage.blocks):
            # block is a SegformerLayer. Swap out block.mlp
            moe_layer = MoELayer(
                hidden_dim=hidden_dim,
                num_experts=moe_config["num_experts"],
                top_k=moe_config["top_k"],
                capacity_factor=moe_config["capacity_factor"],
                noisy_gating=moe_config["noisy_gating"],
                balance_loss_coef=moe_config["balance_loss_coef"]
            )
            block.mlp = moe_layer
            replaced_count += 1
            
    logger.info(f"Successfully replaced {replaced_count} Segformer Feed-Forward blocks with custom MoELayers.")
    return replaced_count
    """, language="python")
    
    st.markdown("#### **2. Gating and Routing Mechanism**")
    st.markdown("This is how the Mixture of Experts layers dispatches individual token vectors to the top-k experts and computes routing weights:")
    st.code("""
class MoELayer(nn.Module):
    # ... initializer setups ...
    
    def forward(self, hidden_states: torch.Tensor, height: int = None, width: int = None) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Reshape to 2D token representation
        tokens = hidden_states.view(-1, hidden_dim) # shape: (num_tokens, hidden_dim)
        
        # Compute routing gating logits
        gating_logits = self.gating(tokens)
        
        # Run router to select top-k indices and compute load balance aux loss
        top_k_gates, top_k_indices, aux_loss = self.router(gating_logits)
        self.aux_loss = aux_loss
        self.last_routing_indices = top_k_indices.detach().cpu()
        
        output = torch.zeros_like(tokens)
        
        for exp_idx, expert in enumerate(self.experts):
            mask = (top_k_indices == exp_idx) # shape: (num_tokens, top_k)
            token_mask = mask.any(dim=-1) # shape: (num_tokens)
            
            if token_mask.any():
                dispatched_tokens = tokens[token_mask]
                expert_outputs = expert(dispatched_tokens)
                
                row_indices, col_indices = torch.where(mask)
                gate_weights = torch.zeros(tokens.size(0), device=tokens.device)
                gate_weights[row_indices] = top_k_gates[row_indices, col_indices]
                
                weighted_outputs = expert_outputs * gate_weights[token_mask].unsqueeze(-1)
                output[token_mask] += weighted_outputs
                
        return output.view(batch_size, seq_len, hidden_dim)
    """, language="python")
