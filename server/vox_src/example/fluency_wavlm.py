import torch
import logging
import sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'fluency'))


from wavlm_fluency import WavLMWrapper

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

    
    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    fluency_label_list = [
        'Fluent', 
        'Disfluent'
    ]

    disfluency_type_labels = [
        "Block", 
        "Prolongation", 
        "Sound Repetition", 
        "Word Repetition", 
        "Interjection"
    ]

    # Define the model
    # Note that ensemble yields the better performance than the single model, but this example is only about wavlm-large
    wavlm_model = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-speech-flow").to(device)
    wavlm_model.eval()

    utterance_fluency_list = list()
    utterance_disfluency_list = list()

    # The way we do inference for fluency is different as the training data is 3s, so we need to do some shifting
    audio_data = torch.zeros([1, 16000*10]).float().to(device)
    audio_segment = (audio_data.shape[1] - 3*16000) // 16000 + 1
    if audio_segment < 1: audio_segment = 1
    input_audio = list()
    for idx in range(audio_segment): input_audio.append(audio_data[0, 16000*idx:16000*idx+3*16000])
    input_audio = torch.stack(input_audio, dim=0)
    
    with torch.no_grad():
        wavlm_fluency_outputs, wavlm_disfluency_outputs = wavlm_model(input_audio)
        fluency_prob   = F.softmax(wavlm_fluency_outputs, dim=1).detach().cpu().numpy().astype(float).tolist()

        disfluency_prob = nn.Sigmoid()(wavlm_disfluency_outputs)
        # we can set a higher threshold in practice
        disfluency_predictions = (disfluency_prob > 0.7).int().detach().cpu().numpy().tolist()
        disfluency_prob = disfluency_prob.cpu().numpy().astype(float).tolist()
        
    # Now lets gather the predictions for the utterance
    for audio_idx in range(audio_segment):
        disfluency_type = list()
        if fluency_prob[audio_idx][0] > 0.5: 
            utterance_fluency_list.append("Fluent")
        else: 
            # If the prediction is disfluent, then which disfluency type
            utterance_fluency_list.append("Disfluent")
            predictions = disfluency_predictions[audio_idx]
            for label_idx in range(len(predictions)):
                if predictions[label_idx] == 1: 
                    disfluency_type.append(disfluency_type_labels[label_idx])
        utterance_disfluency_list.append(disfluency_type)

    # Now print how fluent is the utterance
    print(utterance_fluency_list)
    print(utterance_disfluency_list)
