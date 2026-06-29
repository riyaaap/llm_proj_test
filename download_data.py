from dotenv import load_file, load_dotenv
import os
load_dotenv() 
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("jd5697/wikipedia-biology")
curr_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(curr_dir, "data")
ds.save_to_disk(save_path)
print(f"Dataset successfully saved to: {save_path}")

