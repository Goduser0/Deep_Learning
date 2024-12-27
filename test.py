import argparse
parser = argparse.ArgumentParser(description="Classification")
parser.add_argument("--dataset_ratio", type=str, default="10-shot + PCB_200(150)")
parser.add_argument("--dataset", type=int, default=1)
config = parser.parse_args()
print(config)