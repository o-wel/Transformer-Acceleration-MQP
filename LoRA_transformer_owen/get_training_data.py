import os
import requests
from tokenizer import get_vocab_info

# set up path for logging and saving results
relative_path = os.path.dirname(os.path.relpath(__file__))

def get_training_data(path):

    if not os.path.exists(path):
        os.mkdir(path)
            
    input_file_path = os.path.join(path, 'input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
        print("Wrote tiny shakespeare dataset to " + os.path.join(os.getcwd(), input_file_path))
        

def get_random_web_data(path):
    if not os.path.exists(path):
            os.mkdir(path)
                
    input_file_path = os.path.join(path, 'web_data.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://huggingface.co/datasets/Raziel1234/WebText-1/resolve/main/corpus.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)
        print("Wrote web page data " + os.path.join(os.getcwd(), input_file_path))
    
if __name__ == '__main__':
    
    # set up training data and tokenizer
    download_path = os.path.join(relative_path, "training_data")
    get_training_data(download_path)
    get_random_web_data(download_path)
    
    # open training data
    with open(os.path.join(relative_path, "training_data/input.txt"), "r") as f:
        text = f.read()
    
    vocab_size, chars = get_vocab_info(text)
    print(chars)
    
     # open training data
    with open(os.path.join(relative_path, "training_data/web_data.txt"), "r", encoding="utf-8") as f:
        web_text = f.read()
        
    filtered = "".join(ch for ch in web_text if ch in chars)
    
    # filter new dataset to match old one
    with open(os.path.join(relative_path, "training_data/web_data_clean.txt"), "w") as f:
        f.write(filtered)