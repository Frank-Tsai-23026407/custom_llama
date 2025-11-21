import sys
from tqdm import tqdm
import time

print(f"Stderr isatty: {sys.stderr.isatty()}")
print(f"Stdout isatty: {sys.stdout.isatty()}")

for i in tqdm(range(5), desc="Testing TQDM"):
    time.sleep(0.1)
