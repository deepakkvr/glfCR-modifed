import shutil
import os
import argparse

def save_tif_to_kaggle(tif_path, out_dir="/kaggle/working"):
    if not tif_path:
        return
        
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(tif_path)
    out_path = os.path.join(out_dir, filename)

    try:
        shutil.copy(tif_path, out_path)
        print(f"Saved {filename} to {out_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {tif_path}")
    except Exception as e:
        print(f"Error copying {tif_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download images to Kaggle working directory")
    parser.add_argument('--cloudfree_path', type=str, help='Path to cloudfree image')
    parser.add_argument('--sar_path', type=str, help='Path to SAR image')
    parser.add_argument('--image_path', type=str, help='Path to cloudy image')
    # Accepting extra args (like --model_checkpoint) gracefully by using parse_known_args if needed, 
    # but parse_args is fine if we only pass relevant ones. 
    # To be safe against the full command string which might include other flags, 
    # we can use parse_known_args.
    args, unknown = parser.parse_known_args()
    
    save_tif_to_kaggle(args.cloudfree_path)
    save_tif_to_kaggle(args.sar_path)
    save_tif_to_kaggle(args.image_path)

if __name__ == "__main__":
    main()
