#from dotenv import load_file, load_dotenv
import os
#load_dotenv()
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
# ds = load_dataset("jd5697/wikipedia-biology")
# will load as separate splits instead bc has train/validation/test splits

curr_dir = os.path.dirname(os.path.abspath(__file__))

bio_url = "jd5697/wikipedia-biology"
bio_folder = "data"
cnn_url = "ambrosfitz/cnn-daily-grammar"
cnn_folder = "cnn_data"

def download_dataset(hf_data_url, save_subfolder):
    train_ds = load_dataset(hf_data_url, split="train")
    valid_ds = load_dataset(hf_data_url, split="validation")
    test_ds = load_dataset(hf_data_url, split="test")

    save_path = os.path.join(curr_dir, save_subfolder)

    train_ds.save_to_disk(save_path)
    valid_ds.save_to_disk(save_path)
    test_ds.save_to_disk(save_path)

    print(f"Dataset {hf_data_url} successfully saved to: {save_path}")

# download_dataset(bio_url, bio_folder)
download_dataset(cnn_url, cnn_folder)



















