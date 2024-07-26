import torch
from dataclasses import dataclass

@dataclass
class PPORLElement:
    query_tensor : torch.Tensor # (max_input_length)
    response_tensor : torch.Tensor # (max_output_length)
    scores : torch.Tensor # (max_output_length, vocab_size)
    inf_masks : torch.Tensor # (max_output_length, vocab_size)
    rev_kl : torch.Tensor # (max_output_length)
    t_reward : torch.Tensor #(max_output_length)
    ent_reward : torch.Tensor #(max_output_length)
    reward : torch.Tensor #(max_output_length)
    logprobs : torch.Tensor #(max_output_length)
    w : torch.Tensor #(max_output_length)

    id : str 
    full_label_id : torch.Tensor #(max_original_length)

@dataclass
class PPORLBatch:
    batch_query_tensors : torch.Tensor # (bs, max_input_length)
    batch_response_tensors : torch.Tensor # (bs, max_output_length)
    batch_scores : torch.Tensor # (bs, max_output_length, vocab_size)
    batch_inf_masks : torch.Tensor # (bs, max_output_length, vocab_size)
    batch_rev_kl : torch.Tensor # (bs, max_output_length)
    batch_logprobs : torch.Tensor #(bs, max_output_length)

    batch_t_reward : torch.Tensor #(bs, max_output_length)
    batch_ent_reward : torch.Tensor #(bs, max_output_length)
    batch_reward : torch.Tensor #(bs, max_output_length)
    batch_w : torch.Tensor #(bs, max_output_length)

    batch_ids : list
    batch_full_label_ids : list