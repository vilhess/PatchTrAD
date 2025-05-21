import os
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import subprocess
import shutil

def download_file(raw_url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    response = requests.get(raw_url)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded: {dest_path} ({len(response.content)} bytes)")
    else:
        print(f"❌ Failed to download {raw_url} ({response.status_code})")
        print(f"   ↳ Response: {response.text[:200]}...")


def fetch_github_files(api_url, raw_base_url, local_base_path):
    response = requests.get(api_url)
    if response.status_code != 200:
        print(f"Failed to fetch API listing from {api_url}")
        return

    for item in response.json():
        if item['type'] == 'file':
            raw_url = raw_base_url + item['path']
            local_path = os.path.join(local_base_path, item['path'])
            local_path = local_path.replace("ServerMachineDataset/", "")
            download_file(raw_url, local_path)
        elif item['type'] == 'dir':
            # Recursive call for subdirectories
            sub_api_url = item['url']
            fetch_github_files(sub_api_url, raw_base_url, local_base_path)


def load_and_save(category, filename, dataset, dataset_folder, output_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"{dataset}_{category}.npy"), temp)
    return temp.shape

def processing_smd(dataset_folder="data/smd"):
    os.makedirs(os.path.join(dataset_folder, "processed"), exist_ok=True)
    output_folder = os.path.join(dataset_folder, "processed")
    file_list = os.listdir(os.path.join(dataset_folder, "train"))
    for filename in file_list:
        if filename.endswith('.txt'):
            _ = load_and_save('train', filename, filename.strip('.txt'), dataset_folder, output_folder)
            _ = load_and_save('test', filename, filename.strip('.txt'), dataset_folder, output_folder)
            _ = load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder, output_folder)

def processing_swat(root="data/swat"):
    normal = pd.read_excel(os.path.join(root, 'SWaT_Dataset_Normal_v1.xlsx'))
    normal = normal.iloc[1:, 1:-1].to_numpy()

    scaler = StandardScaler()
    x_normal_scaled = scaler.fit_transform(normal)
    np.save(os.path.join(root, "normal.npy"), x_normal_scaled)

    attack = pd.read_excel(os.path.join(root, 'SWaT_Dataset_Attack_v0.xlsx'))

    labels = attack.iloc[1:, -1] == 'Attack'
    labels = labels.to_numpy().astype(int)

    attack = attack.iloc[1:, 1:-1].to_numpy()
    x_attack_scaled = scaler.transform(attack)

    np.save(os.path.join(root, "attack.npy"), x_attack_scaled)
    np.save(os.path.join(root, "attack_label.npy"), labels)


if __name__ == "__main__":

    # 1. Download NAB + NASA datasets
    simple_files = {
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv": "data/nab/nyc_taxi.csv",
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ec2_request_latency_system_failure.csv": "data/nab/ec2_request_latency_system_failure.csv",
    }
    for url, dest in simple_files.items():
        download_file(url, dest)

    # 2. Download SMD dataset using GitHub API
    smd_api_url = "https://api.github.com/repos/NetManAIOps/OmniAnomaly/contents/ServerMachineDataset"
    smd_raw_base = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/refs/heads/master/"
    fetch_github_files(smd_api_url, smd_raw_base, "data/smd")

    os.makedirs("data/nasa", exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", "patrickfleith/nasa-anomaly-detection-dataset-smap-msl",
        "-p", "data/nasa",
        "--unzip"
    ])

    base_dir = "data/nasa"
    train_src = os.path.join(base_dir, "data/data/train")
    test_src = os.path.join(base_dir, "data/data/test")
    train_dst = os.path.join(base_dir, "train")
    test_dst = os.path.join(base_dir, "test")

    # Move train and test folders
    if os.path.exists(train_src):
        shutil.move(train_src, train_dst)
        print(f"✅ Moved: {train_src} → {train_dst}")

    if os.path.exists(test_src):
        shutil.move(test_src, test_dst)
        print(f"✅ Moved: {test_src} → {test_dst}")

    # Remove the now-empty nested 'data' folder
    nested_data_folder = os.path.join(base_dir, "data")
    if os.path.exists(nested_data_folder):
        shutil.rmtree(nested_data_folder)
        print(f"🗑️ Removed: {nested_data_folder}")

    print(f"Processing SMD")
    processing_smd()

    if os.path.exists("data/swat/SWaT_Dataset_Normal_v1.xlsx") and os.path.exists("data/swat/SWaT_Dataset_Attack_v0.xlsx"):
        print(f'Processing SWAT')
        processing_swat()
    else:
        print(f"SWAT dataset files not found. Please claims the files from https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/")
        print(f"Files needed: SWaT_Dataset_Normal_v1.xlsx, SWaT_Dataset_Attack_v0.xlsx")