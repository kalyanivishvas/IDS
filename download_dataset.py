import os
import requests

# Folder to save datasets
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# URLs for the NSL-KDD dataset
urls = {
    "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt"
}

# Download each file
for filename, url in urls.items():
    file_path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"{filename} downloaded successfully.")
    else:
        print(f"{filename} already exists.")
