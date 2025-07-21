import urllib.request
import gzip
import shutil
import os

# Create data directory if not exists
os.makedirs('data', exist_ok=True)

# Download the file
url = 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz'
download_path = 'data/kddcup.data_10_percent.gz'
extract_path = 'data/kddcup.data_10_percent_corrected'

print("Downloading dataset...")
urllib.request.urlretrieve(url, download_path)

# Extract the gz file
print("Extracting dataset...")
with gzip.open(download_path, 'rb') as f_in:
    with open(extract_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Done! Dataset saved to:", extract_path)
