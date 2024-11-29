import argparse

parser = argparse.ArgumentParser(description="Classification")
parser.add_argument("--img_size", type=int, default=128)
config = parser.parse_args()

print(config)