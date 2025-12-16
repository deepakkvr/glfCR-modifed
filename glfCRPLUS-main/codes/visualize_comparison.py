"""
Comprehensive Visualization: SAR + Cloudy Optical + Cloud-Free + Predicted
Creates a single figure showing the complete cloud removal process
Properly scales all outputs to match input optical image range [0, 10000]
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.windows import Window
import os
from pathlib import Path
import argparse
from net_CR_CrossAttention import CloudRemovalCrossAttention


def normalize_band(band, vmin=None, vmax=None):
    """Normalize band to [0, 1] range for visualization"""
    band = np.asarray(band, dtype=np.float32)
    if vmin is None:
        vmin = np.percentile(band, 2)
    if vmax is None:
        vmax = np.percentile(band, 98)
    normalized = (band - vmin) / (vmax - vmin + 1e-8)
    normalized = np.clip(normalized, 0, 1)
    return normalized


def load_sentinel2_image(image_path):
    """Load Sentinel-2 image (13 bands) from TIFF"""
    with rasterio.open(image_path) as src:
        data = src.read()  # (13, H, W)
    return data.astype(np.float32)


def load_sar_image(sar_path):
    """Load SAR image (2 bands: VV, VH) from TIFF"""
    with rasterio.open(sar_path) as src:
        data = src.read()  # (2, H, W)
    return data.astype(np.float32)


def get_rgb_composite(image_13ch):
    """
    Create RGB composite from 13-channel Sentinel-2 image (in range [0, 10000])
    Uses bands: Red(3), Green(2), Blue(1) -> indices 2, 1, 0
    """
    # Extract RGB bands (0-indexed: B=0, G=1, R=2)
    blue = normalize_band(image_13ch[0])
    green = normalize_band(image_13ch[1])
    red = normalize_band(image_13ch[2])
    
    rgb = np.stack([red, green, blue], axis=2)
    return np.clip(rgb, 0, 1)


def get_sar_rgb(sar_image):
    """
    Create RGB visualization from SAR (2 channels: VV, VH)
    VV (VV) -> Red
    VH (VH) -> Green
    VV+VH -> Blue
    """
    vv = normalize_band(sar_image[0])
    vh = normalize_band(sar_image[1])
    
    rgb = np.stack([
        vv,           # Red
        vh,           # Green
        (vv + vh)/2   # Blue
    ], axis=2)
    return np.clip(rgb, 0, 1)


def get_nir_rgb(image_13ch):
    """
    Create False color composite: NIR-R-G
    NIR(8) -> Red, Red(3) -> Green, Green(2) -> Blue
    Uses indices 7, 2, 1 (0-indexed)
    """
    nir = normalize_band(image_13ch[7])
    red = normalize_band(image_13ch[2])
    green = normalize_band(image_13ch[1])
    
    rgb = np.stack([nir, red, green], axis=2)
    return np.clip(rgb, 0, 1)


def run_inference(optical_img, sar_img, model, device):
    """Run model inference"""
    # Normalize inputs
    optical_norm = optical_img / 10000.0  # Normalize to [0, 1]
    sar_norm = sar_img / 10000.0
    
    # Convert to tensors
    optical_tensor = torch.from_numpy(optical_norm).unsqueeze(0).to(device)  # (1, 13, H, W)
    sar_tensor = torch.from_numpy(sar_norm).unsqueeze(0).to(device)  # (1, 2, H, W)
    
    # Inference
    with torch.no_grad():
        output = model(optical_tensor, sar_tensor)
    
    # Convert back to numpy and scale to [0, 10000]
    predicted = output[0].cpu().numpy()  # (13, H, W) in [0, 1]
    predicted = np.clip(predicted * 10000.0, 0, 10000).astype('float32')
    
    return predicted


def calculate_metrics(predicted, reference):
    """Calculate PSNR and SSIM on images in [0, 10000] range"""
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    
    # Normalize to [0, 1] for metrics
    pred_norm = np.clip(predicted / 10000.0, 0, 1)
    ref_norm = np.clip(reference / 10000.0, 0, 1)
    
    # Calculate metrics for each band and average
    psnr_values = []
    ssim_values = []
    
    for b in range(predicted.shape[0]):
        psnr = peak_signal_noise_ratio(ref_norm[b], pred_norm[b], data_range=1.0)
        ssim = structural_similarity(ref_norm[b], pred_norm[b], data_range=1.0)
        psnr_values.append(psnr)
        ssim_values.append(ssim)
    
    return np.mean(psnr_values), np.mean(ssim_values)


def create_comparison_figure(sar_img, cloudy_opt, cloudfree_opt, predicted_opt, 
                            psnr=None, ssim=None, output_path=None):
    """
    Create comprehensive comparison figure with 4 main views and additional details
    All images in [0, 10000] range for proper comparison
    
    Layout:
    ┌─────────────────────────────────────────────┐
    │  SAR (VV/VH)  │  Cloudy RGB  │  Cloudfree RGB  │  Predicted RGB  │
    │               │              │                 │                 │
    ├─────────────────────────────────────────────┤
    │  SAR (Gray)   │ Cloudy NIR   │  Cloudfree NIR  │  Predicted NIR  │
    │               │              │                 │                 │
    └─────────────────────────────────────────────┘
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: RGB Composites
    # SAR visualization
    ax1 = fig.add_subplot(gs[0, 0])
    sar_rgb = get_sar_rgb(sar_img)
    ax1.imshow(sar_rgb)
    ax1.set_title('SAR (VV/VH Composite)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Cloudy optical RGB
    ax2 = fig.add_subplot(gs[0, 1])
    cloudy_rgb = get_rgb_composite(cloudy_opt)
    ax2.imshow(cloudy_rgb)
    ax2.set_title('Cloudy Optical (RGB)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Cloud-free optical RGB (ground truth)
    ax3 = fig.add_subplot(gs[0, 2])
    cloudfree_rgb = get_rgb_composite(cloudfree_opt)
    ax3.imshow(cloudfree_rgb)
    ax3.set_title('Cloud-Free GT (RGB)', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Predicted RGB
    ax4 = fig.add_subplot(gs[0, 3])
    predicted_rgb = get_rgb_composite(predicted_opt)
    ax4.imshow(predicted_rgb)
    title_text = 'Predicted (RGB)'
    if psnr is not None and ssim is not None:
        title_text += f'\nPSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}'
    ax4.set_title(title_text, fontsize=12, fontweight='bold', color='green')
    ax4.axis('off')
    
    # Row 2: False Color (NIR-R-G)
    # SAR grayscale
    ax5 = fig.add_subplot(gs[1, 0])
    sar_gray = normalize_band(sar_img[0])
    ax5.imshow(sar_gray, cmap='gray')
    ax5.set_title('SAR VV (Grayscale)', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # Cloudy optical NIR
    ax6 = fig.add_subplot(gs[1, 1])
    cloudy_nir = get_nir_rgb(cloudy_opt)
    ax6.imshow(cloudy_nir)
    ax6.set_title('Cloudy NIR-R-G', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    # Cloud-free optical NIR
    ax7 = fig.add_subplot(gs[1, 2])
    cloudfree_nir = get_nir_rgb(cloudfree_opt)
    ax7.imshow(cloudfree_nir)
    ax7.set_title('Cloud-Free NIR-R-G', fontsize=12, fontweight='bold')
    ax7.axis('off')
    
    # Predicted NIR
    ax8 = fig.add_subplot(gs[1, 3])
    predicted_nir = get_nir_rgb(predicted_opt)
    ax8.imshow(predicted_nir)
    ax8.set_title('Predicted NIR-R-G', fontsize=12, fontweight='bold', color='green')
    ax8.axis('off')
    
    # Main title
    fig.suptitle('Cloud Removal: SAR-Guided Optical Restoration\nCrossAttention Model (All images in [0, 10000] range)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison figure saved to: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize cloud removal results')
    parser.add_argument('--optical_path', required=True, help='Path to cloudy optical image')
    parser.add_argument('--sar_path', required=True, help='Path to SAR image')
    parser.add_argument('--cloudfree_path', required=True, help='Path to cloud-free reference image')
    parser.add_argument('--model_checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', default='/kaggle/working/visualization', 
                       help='Output directory for visualization')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device: cuda or cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Loading Data...")
    print("="*60)
    
    # Load images
    print(f"Loading cloudy optical: {args.optical_path}")
    cloudy_opt = load_sentinel2_image(args.optical_path)
    print(f"  Shape: {cloudy_opt.shape}, Range: [{cloudy_opt.min():.1f}, {cloudy_opt.max():.1f}]")
    
    print(f"Loading SAR image: {args.sar_path}")
    sar_img = load_sar_image(args.sar_path)
    print(f"  Shape: {sar_img.shape}, Range: [{sar_img.min():.1f}, {sar_img.max():.1f}]")
    
    print(f"Loading cloud-free reference: {args.cloudfree_path}")
    cloudfree_opt = load_sentinel2_image(args.cloudfree_path)
    print(f"  Shape: {cloudfree_opt.shape}, Range: [{cloudfree_opt.min():.1f}, {cloudfree_opt.max():.1f}]")
    
    print("\n" + "="*60)
    print("Loading Model...")
    print("="*60)
    
    # Load model
    print(f"Loading model checkpoint: {args.model_checkpoint}")
    model = CloudRemovalCrossAttention()
    checkpoint = torch.load(args.model_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("Running Inference...")
    print("="*60)
    
    # Run inference
    print("Processing image...")
    predicted_opt = run_inference(cloudy_opt, sar_img, model, device)
    print(f"✓ Inference complete")
    print(f"  Predicted shape: {predicted_opt.shape}, Range: [{predicted_opt.min():.1f}, {predicted_opt.max():.1f}]")
    
    print("\n" + "="*60)
    print("Calculating Metrics...")
    print("="*60)
    
    # Calculate metrics
    psnr, ssim = calculate_metrics(predicted_opt, cloudfree_opt)
    print(f"✓ PSNR: {psnr:.4f} dB")
    print(f"✓ SSIM: {ssim:.4f}")
    
    print("\n" + "="*60)
    print("Creating Visualization...")
    print("="*60)
    
    # Create comparison figure
    output_fig = os.path.join(args.output_dir, 'cloud_removal_comparison.png')
    create_comparison_figure(sar_img, cloudy_opt, cloudfree_opt, predicted_opt, 
                            psnr=psnr, ssim=ssim, output_path=output_fig)
    
    # Save metrics
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Cloud Removal Results (All images in [0, 10000] range)\n")
        f.write("="*50 + "\n")
        f.write(f"PSNR: {psnr:.4f} dB\n")
        f.write(f"SSIM: {ssim:.4f}\n")
    print(f"✓ Metrics saved to: {metrics_file}")
    
    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print("="*60)


if __name__ == "__main__":
    main()
