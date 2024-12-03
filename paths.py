import os

# The root installation path
SAMPLIFY_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure a directory exists
def ensure_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

# Local copy of data
DATASET_DATA_DIR = os.path.join(SAMPLIFY_DIR, 'datasets', 'data')

# Default output directory
OUTPUT_DIR = os.path.join(SAMPLIFY_DIR, '_output')
ensure_dir(OUTPUT_DIR)


if __name__ == "__main__":
    print(f'SAMPLIFY_DIR    : {SAMPLIFY_DIR}')
    print(f'DATASET_DATA_DIR: {DATASET_DATA_DIR}')
    print(f'OUTPUT_DIR      : {OUTPUT_DIR}')

