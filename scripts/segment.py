import os
import subprocess
import argparse
import cv2
import numpy as np
import shutil
from glob import glob
from tqdm import tqdm

def check_status(img_path, output_folder):
    """
    Checks the status of an image folder.
    Returns:
    - 'complete': Folder exists and has both lungs.
    - 'incomplete': Folder exists but missing files.
    - 'missing': Folder does not exist.
    """
    folder_name = os.path.splitext(os.path.basename(img_path))[0]
    target_dir = os.path.join(output_folder, folder_name)
    
    if not os.path.exists(target_dir):
        return 'missing', target_dir
        
    left = glob(os.path.join(target_dir, "left lung.*"))
    right = glob(os.path.join(target_dir, "right lung.*"))
    
    if len(left) > 0 and len(right) > 0:
        return 'complete', target_dir
    return 'incomplete', target_dir

def run_cxas_with_resume(input_folder, output_folder, force_remake=False):
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for ext in extensions:
        image_files.extend(glob(os.path.join(input_folder, ext)))
        image_files.extend(glob(os.path.join(input_folder, ext.upper())))

    if not image_files:
        print(f"\033[91mNo images found in {input_folder}\033[0m")
        return []

    print(f"\033[94mChecking {len(image_files)} images...\033[0m")
    
    for img_path in tqdm(image_files, desc="CXAS Progress", unit="img"):
        filename = os.path.basename(img_path)
        status, target_path = check_status(img_path, output_folder)
        
        if not force_remake:
            if status == 'complete':
                # Use print above tqdm to avoid UI flickering
                tqdm.write(f"\033[92m[SKIP]\033[0m {filename} exists with complete files. Ready for merge.")
                continue
            elif status == 'incomplete':
                tqdm.write(f"\033[93m[REMAKE]\033[0m {filename} folder is incomplete. Re-processing now...")
                # Optional: clean the incomplete folder first
                shutil.rmtree(target_path)
        else:
            if os.path.exists(target_path):
                tqdm.write(f"\033[93m[FORCE]\033[0m {filename} exists but force_remake is True. Re-processing...")
                shutil.rmtree(target_path)

        try:
            cmd = ["cxas", "-i", img_path, "-o", output_folder, "-s"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            tqdm.write(f"\033[91m[ERROR]\033[0m Failed to process {filename}: {e}")
    
    return image_files

def merge_and_cleanup(output_folder):
    print("\n\033[93mMerging masks into final results...\033[0m")
    subfolders = [f.path for f in os.scandir(output_folder) if f.is_dir()]
    
    if not subfolders:
        print("No subfolders found for merging.")
        return

    for folder in tqdm(subfolders, desc="Merging", unit="folder"):
        folder_name = os.path.basename(folder)
        left_path = glob(os.path.join(folder, "left lung.*"))
        right_path = glob(os.path.join(folder, "right lung.*"))
        
        if left_path and right_path:
            left_img = cv2.imread(left_path[0], cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(right_path[0], cv2.IMREAD_GRAYSCALE)
            
            if left_img is not None and right_img is not None:
                merged = cv2.bitwise_or(left_img, right_img)
                ext = os.path.splitext(left_path[0])[1]
                final_path = os.path.join(output_folder, f"{folder_name}{ext}")
                
                cv2.imwrite(final_path, merged)
                shutil.rmtree(folder)
            else:
                tqdm.write(f"\033[91m[SKIP MERGE]\033[0m Files in {folder_name} are corrupted.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input folder containing original images")
    parser.add_argument("-o", "--output", required=True, help="Output folder for results")
    parser.add_argument("-f", "--force", action="store_true", help="Ignore existing results and remake all")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"\033[91mError: Input path '{args.input}' not found.\033[0m")
        return

    os.makedirs(args.output, exist_ok=True)

    try:
        if run_cxas_with_resume(args.input, args.output, force_remake=args.force):
            merge_and_cleanup(args.output)
            print(f"\n\033[1;32mProcess finished. All results are in: {args.output}\033[0m")
    except KeyboardInterrupt:
        print("\n\033[93mInterrupted by user. Progress saved.\033[0m")
    except Exception as e:
        print(f"\n\033[91mFatal Error: {e}\033[0m")

if __name__ == "__main__":
    main()