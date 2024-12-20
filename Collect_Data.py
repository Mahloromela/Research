import os
import glob
import subprocess

def delete_data(directory="Data"):
    """Delete all data files from the directory after training."""
    files = glob.glob(f"{directory}/*")
    for file in files:
        os.remove(file)
    print("All data files deleted from the directory.")


# Loop through multiple batches
for batch_num in range(10):  # Adjust the range for the number of batches
    print(f"\nProcessing batch {batch_num + 1}")

    # Generate data
    subprocess.run(["python3", r"Simulate.py"])
    delete_data()
