import torch
import sys, os, pdb
import argparse, logging
import torch.nn.functional as F

from pathlib import Path


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'accent'))

from whisper_accent import WhisperWrapper

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 


if __name__ == '__main__':

    english_accent_list = [
        'British Isles', 'North America', 'Other'
    ]
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-broad-accent").to(device)
    model.eval()
    
    # Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
    # So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
    data = torch.zeros([1, 16000]).float().to(device)
    logits, embeddings = model(data, return_feature=True)
    
    # Probability
    accent_prob = F.softmax(logits, dim=1)
    print(english_accent_list[torch.argmax(accent_prob).detach().cpu().item()])
    