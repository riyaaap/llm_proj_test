import os
from datasets import load_from_disk
curr_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(curr_dir, "data")

dataset = load_from_disk(save_dir)
print(dataset)
print(dataset[0])
print(dataset[0:15])


