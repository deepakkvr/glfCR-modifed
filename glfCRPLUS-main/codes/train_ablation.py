"""
Ablation Study Training Script: SAR Noise-Handling Strategies

This script trains ablation variants of the cloud removal model to compare:
1. Original: Speckle-aware gating module
2. NoDenoising: Direct cross-attention output (no noise handling)
3. FullDenoising: Explicit SAR denoising before cross-attention

Key features:
- Loads pretrained checkpoint from original model
- Freezes shared modules (encoders, cross-attention, decoder)
- Only trains noise-handling and refinement modules
- Supports variant selection via command-line argument
"""

import os
import sys
import torch
import torch.nn as nn
import argparse
import numpy as np
import json
from datetime import datetime
import time

# Enable cuDNN autotuner for optimal performance
torch.backends.cudnn.benchmark = True

from dataloader import *
from metrics import *

# Import all model variants
from net_CR_CrossAttention import CloudRemovalCrossAttention
from net_CR_NoDenoising import CloudRemovalCrossAttentionNoDenoising
from net_CR_FullDenoising import CloudRemovalCrossAttentionFullDenoising

# Try to import enhancements
try:
    from enhancements.losses import EnhancedLoss
except ImportError:
    print("Warning: enhancements.losses not found. Using standard L1 loss.")
    EnhancedLoss = None


##===================================================##
##********** Configure training settings ************##
##===================================================##
parser = argparse.ArgumentParser(description='Ablation Study Training Script')

# Ablation variant selection
parser.add_argument('--variant', type=str, required=True,
                    choices=['original', 'no_denoising', 'full_denoising'],
                    help='Ablation variant to train: original (with gating), no_denoising (no noise handling), full_denoising (explicit denoiser)')

# Checkpoint and data paths
parser.add_argument('--pretrained_path', type=str, required=True,
                    help='Path to pretrained original model checkpoint')
parser.add_argument('--input_data_folder', type=str, default='../data')
parser.add_argument('--data_list_filepath', type=str, default='../data/data.csv')

# Training hyperparameters
parser.add_argument('--batch_sz', type=int, default=4, help='batch size used for training')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--load_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=128)

parser.add_argument('--optimizer', type=str, default='Adam', help='Adam optimizer')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (reduced for fine-tuning)')
parser.add_argument('--lr_step', type=int, default=5, help='lr decay step')
parser.add_argument('--lr_start_epoch_decay', type=int, default=10, help='epoch to start lr decay')
parser.add_argument('--max_epochs', type=int, default=15, help='maximum training epochs')
parser.add_argument('--save_freq', type=int, default=5, help='save checkpoint every N epochs')
parser.add_argument('--save_model_dir', type=str, default='./checkpoints/ablation', help='checkpoint directory')

# Optional settings
parser.add_argument('--is_use_cloudmask', type=bool, default=False)
parser.add_argument('--cloud_threshold', type=float, default=0.2)
parser.add_argument('--is_test', type=bool, default=False, help='whether in test mode')
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--dry_run', action='store_true', help='Test checkpoint loading without full training')

opts = parser.parse_args()


##===================================================##
##************** Helper Functions *******************##
##===================================================##
def get_model(variant):
    """Get the appropriate model based on variant selection"""
    if variant == 'original':
        return CloudRemovalCrossAttention()
    elif variant == 'no_denoising':
        return CloudRemovalCrossAttentionNoDenoising()
    elif variant == 'full_denoising':
        return CloudRemovalCrossAttentionFullDenoising()
    else:
        raise ValueError(f"Unknown variant: {variant}")


def freeze_modules(model, variant):
    """
    Freeze shared modules (encoders, cross-attention, decoder).
    Only allow training for noise-handling and refinement modules.
    """
    # Freeze optical encoder
    for param in model.optical_encoder.parameters():
        param.requires_grad = False
    
    # Freeze SAR encoder
    for param in model.sar_encoder.parameters():
        param.requires_grad = False
    
    # Freeze cross-attention
    for param in model.cross_attn.parameters():
        param.requires_grad = False
    
    # Freeze decoder
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    # Refinement block is always trainable (for all variants)
    for param in model.refinement.parameters():
        param.requires_grad = True
    
    # Variant-specific trainable modules
    if variant == 'original':
        # Allow training of speckle-aware gating module
        for param in model.speckle_gating.parameters():
            param.requires_grad = True
        print("✓ Trainable modules: speckle_gating, refinement")
    elif variant == 'no_denoising':
        # No additional modules to train (only refinement)
        print("✓ Trainable modules: refinement only")
    elif variant == 'full_denoising':
        # Allow training of SAR denoiser
        for param in model.sar_denoiser.parameters():
            param.requires_grad = True
        print("✓ Trainable modules: sar_denoiser, refinement")


def load_pretrained_checkpoint(model, checkpoint_path, variant):
    """
    Load pretrained checkpoint with selective loading.
    Only load weights for shared modules (encoders, cross-attention, decoder).
    Initialize new modules (denoiser/gating) with random weights.
    """
    print(f"\nLoading pretrained checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        pretrained_state = checkpoint['model_state_dict']
    else:
        pretrained_state = checkpoint
    
    # Get current model state
    model_state = model.state_dict()
    
    # Selective loading: only load compatible keys
    loaded_keys = []
    skipped_keys = []
    
    for key, value in pretrained_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
    
    # Load the updated state dict
    model.load_state_dict(model_state)
    
    print(f"✓ Loaded {len(loaded_keys)} parameter tensors from checkpoint")
    if skipped_keys:
        print(f"⚠ Skipped {len(skipped_keys)} incompatible keys (will be initialized randomly)")
        if variant == 'full_denoising':
            print("  (This is expected for 'full_denoising' variant - sar_denoiser is new)")
    
    return model


def get_trainable_params(model):
    """Count trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_pct': 100.0 * trainable_params / total_params
    }


def validate(model, val_dataloader, device):
    """Validate model on validation set"""
    model.eval()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in val_dataloader:
            # Handle both old and new key names
            cloudy_key = 'cloudy_optical' if 'cloudy_optical' in data else 'cloudy_data'
            sar_key = 'sar' if 'sar' in data else 'SAR_data'
            cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in data else 'cloudfree_data'
            
            cloudy_optical = data[cloudy_key].to(device)
            sar_img = data[sar_key].to(device)
            cloudfree_data = data[cloudfree_key].to(device)
            
            pred = model(cloudy_optical, sar_img)
            
            batch_psnr = PSNR(pred, cloudfree_data)
            batch_ssim = SSIM(pred, cloudfree_data)
            
            total_psnr += batch_psnr
            total_ssim += batch_ssim
            num_batches += 1
    
    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    
    model.train()
    return avg_psnr, avg_ssim


def save_checkpoint(model, optimizer, lr_scheduler, epoch, val_psnr, best_val_psnr, opts):
    """Save training checkpoint"""
    os.makedirs(opts.save_model_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'variant': opts.variant,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'val_psnr': val_psnr,
        'best_val_psnr': best_val_psnr,
        'opts': opts
    }
    
    # Save regular checkpoint with variant name
    checkpoint_path = os.path.join(opts.save_model_dir, f'ablation_{opts.variant}_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save best model
    is_best = val_psnr > best_val_psnr
    if is_best:
        best_path = os.path.join(opts.save_model_dir, f'ablation_{opts.variant}_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path} (PSNR: {val_psnr:.2f} dB)")
    
    return is_best


def seed_torch(seed=7):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


##===================================================##
##******************** Main *************************##
##===================================================##
if __name__ == '__main__':
    ##===================================================##
    ##*************** Print configuration ***************##
    ##===================================================##
    print("\n" + "="*60)
    print(f"ABLATION STUDY: {opts.variant.upper()} VARIANT")
    print("="*60)
    for arg in vars(opts):
        print(f"{arg:.<30} {getattr(opts, arg)}")
    print("="*60 + "\n")

    ##===================================================##
    ##****************** Choose GPU *********************##
    ##===================================================##
    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    ##===================================================##
    ##*************** Create dataloader *****************##
    ##===================================================##
    seed_torch()

    train_filelist, val_filelist, _ = get_train_val_test_filelists(opts.data_list_filepath)

    print(f"Training samples: {len(train_filelist)}")
    print(f"Validation samples: {len(val_filelist)}\n")

    # Training dataloader with optimizations
    train_data = AlignedDataset(opts, train_filelist)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_data,
        batch_size=opts.batch_sz,
        shuffle=True,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Validation dataloader
    val_data = AlignedDataset(opts, val_filelist)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_data,
        batch_size=opts.batch_sz,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    ##===================================================##
    ##****************** Create model *******************##
    ##===================================================##
    print("="*60)
    print("Loading Model and Checkpoint")
    print("="*60)
    
    # Create model based on variant
    model = get_model(opts.variant)
    
    # Load pretrained checkpoint (selective loading)
    model = load_pretrained_checkpoint(model, opts.pretrained_path, opts.variant)
    
    # Move model to device
    model = model.to(device)
    
    # Freeze modules
    print("\nFreezing shared modules...")
    freeze_modules(model, opts.variant)
    
    # Print parameter statistics
    param_stats = get_trainable_params(model)
    print(f"\nParameter Statistics:")
    print(f"  Total parameters:     {param_stats['total']:,}")
    print(f"  Trainable parameters: {param_stats['trainable']:,} ({param_stats['trainable_pct']:.2f}%)")
    print(f"  Frozen parameters:    {param_stats['frozen']:,}")
    print("="*60 + "\n")
    
    # Dry run mode: just test checkpoint loading
    if opts.dry_run:
        print("✓ Dry run successful! Checkpoint loaded and modules frozen correctly.")
        print("Exiting without training.\n")
        sys.exit(0)

    ##===================================================##
    ##************** Setup optimizer ********************##
    ##===================================================##
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam(trainable_params, lr=opts.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=opts.lr_step, gamma=0.5
    )
    
    # Loss function
    criterion = nn.L1Loss()

    ##===================================================##
    ##**************** Train the network ****************##
    ##===================================================##
    print("\n" + "="*60)
    print("Starting Ablation Training")
    print("="*60 + "\n")

    # Initialize logging
    training_log = {
        'variant': opts.variant,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(opts),
        'epochs': []
    }
    
    log_dir = os.path.join(opts.save_model_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'ablation_{opts.variant}_log.json')

    best_val_psnr = 0.0
    train_start_time = time.time()

    for epoch in range(opts.max_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{opts.max_epochs-1} - Variant: {opts.variant}")
        print(f"{'='*60}")
        
        model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, data in enumerate(train_dataloader):
            # Handle both old and new key names
            cloudy_key = 'cloudy_optical' if 'cloudy_optical' in data else 'cloudy_data'
            sar_key = 'sar' if 'sar' in data else 'SAR_data'
            cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in data else 'cloudfree_data'
            
            cloudy_optical = data[cloudy_key].to(device)
            sar_img = data[sar_key].to(device)
            cloudfree_data = data[cloudfree_key].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred = model(cloudy_optical, sar_img)
            loss = criterion(pred, cloudfree_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_dataloader):
                progress_pct = 100 * (batch_idx + 1) / len(train_dataloader)
                bar_length = 30
                filled = int(bar_length * (batch_idx + 1) / len(train_dataloader))
                bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
                print(f"\r  [{bar}] {progress_pct:5.1f}% ({batch_idx+1}/{len(train_dataloader)}) - Loss: {loss.item():.4f}", end='', flush=True)
        
        # Epoch statistics
        print()  # New line after progress bar
        avg_train_loss = epoch_loss / num_batches
        
        # Calculate training PSNR on last batch
        with torch.no_grad():
            avg_train_psnr = PSNR(pred, cloudfree_data)
        
        # Validation
        print(f"\nRunning validation...")
        val_psnr, val_ssim = validate(model, val_dataloader, device)
        
        # Update learning rate
        if epoch >= opts.lr_start_epoch_decay:
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate updated: {current_lr:.6f}")
        
        # Check if best model
        is_best = val_psnr > best_val_psnr
        if is_best:
            best_val_psnr = val_psnr
        
        # Save checkpoint at configured frequency and always on last epoch
        if (epoch % opts.save_freq == 0) or (epoch == opts.max_epochs - 1):
            save_checkpoint(model, optimizer, lr_scheduler, epoch, val_psnr, best_val_psnr, opts)
        
        epoch_time = time.time() - epoch_start_time
        
        # Log epoch results
        epoch_log = {
            'epoch': epoch,
            'train_loss': float(avg_train_loss),
            'train_psnr': float(avg_train_psnr),
            'val_psnr': float(val_psnr),
            'val_ssim': float(val_ssim),
            'best_val_psnr': float(best_val_psnr),
            'learning_rate': float(optimizer.param_groups[0]['lr']),
            'epoch_time': float(epoch_time),
            'is_best': is_best
        }
        training_log['epochs'].append(epoch_log)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train PSNR: {avg_train_psnr:.2f} dB")
        print(f"  Val PSNR:   {val_psnr:.2f} dB")
        print(f"  Val SSIM:   {val_ssim:.4f}")
        print(f"  Best Val PSNR: {best_val_psnr:.2f} dB {'(NEW!)' if is_best else ''}")
        print(f"  Epoch Time: {epoch_time/60:.1f} minutes")
        print(f"{'='*60}")
        
        # Save training log after each epoch
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

    # Training complete
    total_time = time.time() - train_start_time
    training_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_log['total_time_hours'] = total_time / 3600

    # Final log save
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Ablation Training Complete - Variant: {opts.variant}")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Training log saved to: {log_path}")
    print(f"Best model saved to: {os.path.join(opts.save_model_dir, f'ablation_{opts.variant}_best.pth')}")
    print(f"{'='*60}\n")
