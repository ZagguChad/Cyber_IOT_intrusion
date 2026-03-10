# ======================================
# LOCAL FILE LISTING
# ======================================
import os

DATA_DIR = r"D:\cyberr\wustl_iiot_2021"

for dirname, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        filepath = os.path.join(dirname, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"{filepath}  ({size_mb:.1f} MB)")
