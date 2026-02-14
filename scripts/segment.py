import os
import subprocess
import argparse
import cv2
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm

def run_cxas_batch(input_folder, output_folder):
    # Support common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_folder, ext)))
        image_files.extend(glob(os.path.join(input_folder, ext.upper())))

    if not image_files:
        print(f"\033[91mNo images found in {input_folder}\033[0m")
        return []

    print(f"\033[94mFound {len(image_files)} images. Starting CXAS...\033[0m")
    
    for img_path in tqdm(image_files, desc="Running CXAS", unit="img"):
        cmd = ["cxas", "-i", img_path, "-o", output_folder, "-s"]
        # Mute stdout to keep tqdm clean
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    return image_files

def process_and_cleanup(output_folder):
    print("\n\033[93mMerging masks and cleaning up...\033[0m")
    
    # Get all subfolders (e.g., tmp/output/JPCLN014/)
    subfolders = [f.path for f in os.scandir(output_folder) if f.is_dir()]
    
    for folder in tqdm(subfolders, desc="Merging Results", unit="folder"):
        folder_name = os.path.basename(folder)
        
        # Search for lung files inside the subfolder
        left_path = glob(os.path.join(folder, "left lung.*"))
        right_path = glob(os.path.join(folder, "right lung.*"))
        
        if left_path and right_path:
            # Read images
            left_img = cv2.imread(left_path[0], cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(right_path[0], cv2.IMREAD_GRAYSCALE)
            
            # Combine binary masks
            merged = cv2.bitwise_or(left_img, right_img)
            
            # Save as output_folder/JPCLN014_cxas_pred.png
            ext = os.path.splitext(left_path[0])[1]
            final_filename = f"{folder_name}_cxas_pred{ext}"
            final_path = os.path.join(output_folder, final_filename)
            
            cv2.imwrite(final_path, merged)
            
            # Delete the subfolder and its contents after successful merge
            shutil.rmtree(folder)

def interactive_prompt():
    print("\033[1;36m" + "="*35)
    print("  CXAS WRAPPER & MERGER  ")
    print("="*35 + "\033[0m")
    in_path = input("\033[92mInput Folder  : \033[0m").strip()
    out_path = input("\033[92mOutput Folder : \033[0m").strip()
    return in_path, out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Input image folder")
    parser.add_argument("-o", "--output", help="Output results folder")
    args = parser.parse_args()

    input_dir, output_dir = (args.input, args.output) if args.input and args.output else interactive_prompt()

    if not os.path.exists(input_dir):
        print(f"\033[91mError: Input path not found.\033[0m")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        if run_cxas_batch(input_dir, output_dir):
            process_and_cleanup(output_dir)
            print("\n\033[1;32mDone! Results saved in:\033[0m", output_dir)
    except Exception as e:
        print(f"\n\033[91mError occurred: {e}\033[0m")

if __name__ == "__main__":
    main()