import os
from datasets import load_from_disk
curr_dir = os.path.dirname(os.path.abspath(__file__))

biofolder = "data"
cnnfolder = "cnn_data"

def view(data_folder):
    save_dir = os.path.join(curr_dir, data_folder)

    dataset = load_from_disk(save_dir)
    print(dataset)
    print(dataset[0])
    print(dataset[0:15])


view(biofolder)
view(cnnfolder)
