import gdown
import os

os.makedirs("models", exist_ok=True)

# Read file_id and filename pairs
with open("file_ids.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Download each file with the specified name
for line in lines:
    parts = line.split(",", 1)
    if len(parts) != 2:
        print(f"Invalid line format: {line}")
        continue

    file_id, filename = parts
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join("models", filename)

    print(f"Downloading {filename}...")
    gdown.download(url, output_path, quiet=False)
