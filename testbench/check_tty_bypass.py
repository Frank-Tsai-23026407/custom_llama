import sys
from tqdm import tqdm
import time

try:
    with open('/dev/tty', 'w') as tty:
        print("Writing to /dev/tty directly", file=tty)
        for i in tqdm(range(5), desc="TQDM to TTY", file=tty):
            time.sleep(0.1)
except Exception as e:
    print(f"Failed to write to /dev/tty: {e}")

print("Standard print to stdout")
