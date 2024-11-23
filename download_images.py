
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Function to get the desktop path dynamically
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"), "Desktop")

# Directory to save images
output_dir = os.path.join(get_desktop_path(), "faces")
os.makedirs(output_dir, exist_ok=True)

# Function to download a single image
def download_single_image(base_url, index):
    try:
        response = requests.get(base_url, stream=True, timeout=10)  # Timeout to prevent hanging
        if response.status_code == 200:
            # Save the image
            with open(f"{output_dir}/face_{index:05d}.jpg", "wb") as f:
                f.write(response.content)
        else:
            print(f"Failed to download image {index}: Status {response.status_code}")
    except Exception as e:
        print(f"Error downloading image {index}: {e}")

# Function to download images concurrently
def download_images(base_url, num_images, max_threads=20):
    with ThreadPoolExecutor(max_threads) as executor:
        list(tqdm(executor.map(lambda i: download_single_image(base_url, i), range(num_images)), 
                  total=num_images, 
                  desc="Downloading images"))

# Replace with the actual URL of the image source
base_url = "https://thispersondoesnotexist.com/"
num_images = 50000  # Number of images to download
max_threads = 30  # Number of threads to use for downloading

# Start the download process
download_images(base_url, num_images, max_threads)

print(f"Download completed. Images are saved in {output_dir}")
