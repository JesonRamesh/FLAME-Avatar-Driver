import os
import urllib.request

# Create mappings directory if it doesn't exist
os.makedirs('mappings', exist_ok=True)

# URLs for the pre-trained mapping files
files = {
    'bs2exp.npy': 'https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame/raw/main/mappings/bs2exp.npy',
    'bs2pose.npy': 'https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame/raw/main/mappings/bs2pose.npy',
    'bs2eye.npy': 'https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame/raw/main/mappings/bs2eye.npy'
}

print("Downloading pre-trained MediaPipe → FLAME mappings...")
print("From: https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame\n")

for filename, url in files.items():
    filepath = f'mappings/{filename}'
    
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists, skipping...")
        continue
    
    try:
        print(f"  Downloading {filename}...", end=' ')
        urllib.request.urlretrieve(url, filepath)
        print("✓ Done")
    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n=== Download Complete ===")
print("Run your main script again to use the pre-trained mappings!")
print("You should see: '✓ Using pre-trained mappings from PeizhiYan's repository'")