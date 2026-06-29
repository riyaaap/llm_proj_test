#from dotenv import load_file, load_dotenv
import os
#load_dotenv()
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("jd5697/wikipedia-biology")
# will load as separate splits instead bc has train/validation/test splits 
curr_dir = os.path.dirname(os.path.abspath(__file__))

train_ds = load_dataset("jd5697/wikipedia-biology", split="train")
valid_ds = load_dataset("jd5697/wikipedia-biology", split="validation")
test_ds = load_dataset("jd5697/wikipedia-biology", split="test")

save_path = os.path.join(curr_dir, "data")

train_ds.save_to_disk(save_path)
valid_ds.save_to_disk(save_path)
test_ds.save_to_disk(save_path)

print(f"Dataset successfully saved to: {save_path}")

