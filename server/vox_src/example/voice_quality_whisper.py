import torch
import sys, os
import torch.nn as nn
from pathlib import Path


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'voice_quality'))

from whisper_voice_quality import WhisperWrapper


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
        'shrill', 'nasal', 'deep',  # Pitch
        'silky', 'husky', 'raspy', 'guttural', 'vocal-fry', # Texture
        'booming', 'authoritative', 'loud', 'hushed', 'soft', # Volume
        'crisp', 'slurred', 'lisp', 'stammering', # Clarity
        'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated',  'hesitant', # Rhythm
    ]
    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    # Define the model
    # Note that ensemble yields the better performance than the single model
    model = WhisperWrapper.from_pretrained("tiantiaf/whisper-large-v3-voice-quality").to(device)
    model.eval()

    # Our training data filters output audio shorter than 3 seconds (unreliable predictions) and longer than 15 seconds (computation limitation)
    # So you need to prepare your audio to a maximum of 15 seconds, 16kHz and mono channel
    data = torch.zeros([1, 16000]).float().to(device)
    logits = model(data, return_feature=False)
    voice_quality_prob = nn.Sigmoid()(torch.tensor(logits))
    
    # In practice, a larger threshold would remove some noise, but it is best to aggregate predictions per speaker
    # For the human evaluating experiments, we set threshold of higher than 0.5
    threshold = 0.5
    predictions = (voice_quality_prob > threshold).int().detach().cpu().numpy()[0].tolist()
    print(voice_quality_prob.shape)
    print(voice_quality_prob)

