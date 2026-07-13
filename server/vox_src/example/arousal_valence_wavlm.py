import torch
import logging
import sys, os, pdb

from pathlib import Path

sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model', 'emotion'))

from wavlm_emotion_dim import WavLMWrapper


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

    # Define the model
    # Note that ensemble yields the better performance than the single model
    # Define the model wrapper
    model_path = "model"
    wavlm_model = WavLMWrapper(
        pretrain_model="wavlm_large", 
        finetune_method="finetune",
        output_class_num=9,
        freeze_params=True, 
        use_conv_output=True,
        detailed_class_num=17
    ).to(device)
    
    wavlm_model.load_state_dict(torch.load(os.path.join(model_path, f"wavlm_arousal_valence.pt"), weights_only=True), strict=False)
    wavlm_model.eval()
    
    # Audio must be 16k Hz
    data = torch.zeros([1, 16000]).to(device)
    wavlm_arousal, wavlm_valence, _      = wavlm_model(data)
    
    print(wavlm_arousal)
    print(wavlm_valence)
