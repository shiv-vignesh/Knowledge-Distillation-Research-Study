import torch, os, pickle
import numpy as np

import torch.nn.functional as F

from transformers import AutoModelForCausalLM, GPT2LMHeadModel

from trainer.logger import Logger

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    
    return logits

def batch_top_k_top_p_filtering(
    logits: torch.tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def filter_top_percent(logits:torch.tensor, percent:float=0.05):

    percent = percent if percent != -1 else 0.05

    _, indices = torch.sort(logits, descending=True)
    cutoff_index = int(percent * logits.shape[-1])
    mask = torch.zeros_like(
        logits
    ).scatter_(-1, indices[:, :cutoff_index], 1.0)

    return logits * mask

def save_tensor2pickle(logits:torch.tensor, fn:str):
    logits = logits.detach().cpu().numpy()
    
    with open(fn, 'wb') as f:
        pickle.dump(logits, f)

def save_tensor2npy(logits:torch.tensor, fn:str):
    logits = logits.detach().cpu().numpy()
    np.save(fn, logits)

def check_gpu_availability():
    return torch.cuda.is_available()

def get_gpu_count():
    return torch.cuda.device_count()

def load_from_ckpt(model:torch.nn.Module, model_ckpt_path:str, device:torch.device, logger:Logger):

    if os.path.exists(model_ckpt_path):
        logger.log_message('Loading saved model ckpt')
        logger.log_new_line()
        state_dict = torch.load(model_ckpt_path, map_location=torch.device(device))
        '''Saved using dataparallel. state_dict key mismatch'''
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict) 

    else:
        logger.log_message(f'Model path unavailable')
        logger.log_new_line()

    model.to(device)       

    return model

def create_logger(output_dir:str):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return Logger(output_dir)