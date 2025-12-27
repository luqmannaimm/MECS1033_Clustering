import os
import shutil
import kagglehub

def main():

    # Download dataset if not exist
    if not os.path.exists("dataset"):
        print("Downloading dataset...")
        temp_path = kagglehub.dataset_download("habbas11/dms-driver-monitoring-system")
        shutil.move(temp_path, ".")
        os.rename("1", "dataset")
    else:
        print("Dataset already exists...")

if __name__ == "__main__":
    main()