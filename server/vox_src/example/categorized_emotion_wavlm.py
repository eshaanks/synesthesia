import torch
import logging
import sys, os, pdb
import torch.nn.functional as F

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'emotion'))

from wavlm_emotion import WavLMWrapper


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

    label_list = [
        'Anger', 
        'Contempt', 
        'Disgust', 
        'Fear', 
        'Happiness', 
        'Neutral', 
        'Sadness', 
        'Surprise', 
        'Other'
    ]

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble yields the better performance than the single model
    wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device)
    wavlm_model.eval()
    
    # Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
    # So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
    data = torch.zeros([1, 16000]).float().to(device)
    wavlm_logits, wavlm_embedding, _, _, _, _ = wavlm_model(
        data, return_feature=True
    )
    
    emotion_prob   = F.softmax(wavlm_logits, dim=1)

    print(emotion_prob.shape)
    print(wavlm_embedding.shape)
    print(label_list[torch.argmax(emotion_prob).detach().cpu().item()])
    pdb.set_trace()

