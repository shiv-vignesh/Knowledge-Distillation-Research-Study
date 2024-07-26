import torch 
from dataset_utils.data_types import PPORLElement, PPORLBatch

def compute_log_probs(logits:torch.tensor, ids:torch.tensor, inf_mask=None):
    logits = torch.nn.functional.log_softmax(
        logits, dim=-1 
    )
    
    # inf_mask = torch.isinf(logits)
    
    if inf_mask is not None:
        logits = torch.masked_fill(
            logits, inf_mask, value=-float('inf')
        )
        
    # else:
    #     pass
    
    # nan_mask = torch.isnan(logits)
    # logits = torch.masked_fill(
    #     logits, nan_mask, value=-float('inf')
    # )
    
    logits = torch.gather(logits, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)
    
    try:
        assert all((~torch.isinf(logits.view(-1))) & (~torch.isnan(logits.view(-1))))
    except:
        print(logits)
        print(logits.size())
        print(f'Assertion Failed!!!')
        exit(1)

    return logits

def compute_advantages_and_returns(
        rewards:torch.tensor,
        mask:torch.tensor,
        gamma:float=0.9
):
    
    rewards = rewards.float()
    mask = mask.float()

    lens = torch.cumsum(mask, dim=-1)
    lens = mask - lens + lens[:, -1:None]
    lens = torch.masked_fill(lens, lens==0,1)

    rewards_reversed = []
    last_rw = 0

    response_length = rewards.size(1)

    for i in reversed(range(response_length)):
        rw_delta = rewards[:, i]
        last_rw = rw_delta + gamma * last_rw
        rewards_reversed.append(last_rw)

    rw = torch.stack(rewards_reversed[::-1], dim=1)
    rw /= lens

    ''' TODO, perform whiten()'''

    return rw 

def get_x_entropy(ppo_logits:torch.tensor, teacher_logits:torch.tensor, 
                  inf_mask:torch.tensor, mask:torch.tensor):
    
    full_probs = torch.nn.functional.softmax(
        ppo_logits, dim=-1, dtype=torch.float32
    )

    full_logprobs = torch.nn.functional.log_softmax(
        teacher_logits, dim=-1, dtype=torch.float32
    )

    if inf_mask is not None:
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
    
    else:
        inf_mask = torch.isinf(full_logprobs)
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
    
    xent = -torch.sum(full_probs * full_logprobs, dim=-1)
    xent = xent * mask

    return xent

def get_entropy(ppo_logits:torch.tensor, inf_mask:torch.tensor, mask:torch.tensor):

    full_probs = torch.nn.functional.softmax(ppo_logits, dim=-1, dtype=torch.float32)
    full_logprobs = torch.nn.functional.log_softmax(ppo_logits, dim=-1, dtype=torch.float32)

    if inf_mask is not None:
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
    else:
        inf_mask = torch.isinf(full_logprobs)
        full_logprobs = full_logprobs.masked_fill(inf_mask, 0)
    
    ent = -torch.sum(full_probs * full_logprobs, dim=-1)

    return ent * mask

def compute_cumsum_rewards(rewards:torch.tensor, gamma:float=0.9):
    full_rewards = torch.zeros_like(rewards[:, 0])

    for i in reversed(range(rewards.size(1))):
        full_rewards = gamma * full_rewards + rewards[:, i]

    return full_rewards

class Losses:

    def cross_entropy_loss(logits:torch.tensor, 
                           labels:torch.tensor,
                           bool_mask:torch.tensor
                           ):
        
        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        masked_loss = loss * bool_mask.view(-1)
        return masked_loss.sum()/bool_mask.sum()
    
    def forward_kl(teacher_logits:torch.tensor,
                   student_logits:torch.tensor,
                   bool_mask:torch.tensor,
                   layer_norm:bool=True,
                   temperature:float=0.7
                   ):
        
        if layer_norm:
            teacher_logits = torch.nn.functional.layer_norm(teacher_logits, teacher_logits.size()[1:])
            student_logits = torch.nn.functional.layer_norm(student_logits, student_logits.size()[1:])

        teacher_logits = torch.nn.functional.softmax(teacher_logits/temperature, dim=-1)
        student_logits = torch.nn.functional.softmax(student_logits/temperature, dim=-1)

        kl_forward = torch.nn.KLDivLoss(reduction='none')
        kl_forward_loss = kl_forward(
            student_logits.log(), teacher_logits
        )
        masked_kl_forward_loss = kl_forward_loss * bool_mask.unsqueeze(-1)

        return masked_kl_forward_loss.sum()/bool_mask.sum()

    def compute_pg_loss(logprobs:torch.tensor,
                        old_logprobs:torch.tensor,
                        advantages:torch.tensor,
                        mask:torch.tensor,
                        w:torch.tensor,
                        cliprange):
        
        n = mask.sum()

        log_ratio = (logprobs - old_logprobs) * mask
        print(log_ratio)
        ratio = torch.exp(log_ratio.float())
        print(ratio)
        ratio = ratio * w

        # ratio = torch.clamp(ratio, -cliprange, cliprange)

        # print(ratio[0])
        # print(log_ratio[0])

        # exit(1)

        assert any(~torch.isinf(ratio.view(-1))) & any(~torch.isnan(ratio.view(-1)))
        assert any(~torch.isinf(advantages.view(-1))) & any(~torch.isnan(advantages.view(-1)))

        # if any(torch.isinf(advantages).view(-1)):
        #     print("[ERROR] advantage inf")
        
        # if any(torch.isinf(ratio).view(-1)):
        #     print("[ERROR] ratio inf")

        # if any(torch.isnan(advantages).view(-1)):
        #     print("[ERROR] advantage nan")
        
        # if any(torch.isnan(ratio).view(-1)):
        #     print("[ERROR] ratio nan")

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - cliprange,
            1.0 + cliprange
        )

        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2).float() * mask) / n 

        return pg_loss

    def reg_loss(teacher_logits:torch.tensor, ppo_logits:torch.tensor, 
                 ppo_rl_batch:PPORLBatch, mask:torch.tensor
                 ):
        
        loss_exp_ent = 0 
        xent = get_x_entropy(ppo_logits=ppo_logits, teacher_logits=teacher_logits, 
                             inf_mask=ppo_rl_batch.batch_inf_masks, mask=mask)
        
        ent = get_entropy(ppo_logits=ppo_logits, inf_mask=ppo_rl_batch.batch_inf_masks, 
                          mask=mask)

        loss_exp_ent = torch.sum((xent - ent) * mask)/mask.sum()
        
        return loss_exp_ent


    def compute_ppo_loss(ppo_logits:torch.tensor, teacher_logits:torch.tensor, 
                         ppo_rl_batch:PPORLBatch, temperature, pad_token_id):

        ppo_logits = ppo_logits / temperature
        
        start = ppo_rl_batch.batch_query_tensors.size(1) - 1 
        end = ppo_rl_batch.batch_query_tensors.size(1) + ppo_rl_batch.batch_response_tensors.size(1) - 1 

        ppo_logits = ppo_logits[:, start:end]

        if ppo_rl_batch.batch_inf_masks is not None:
            ppo_logits = torch.masked_fill(ppo_logits, 
                                           ppo_rl_batch.batch_inf_masks, -float('inf'))
        # else:
        #     inf_mask = torch.isinf(ppo_logits)
        #     ppo_logits = torch.masked_fill(ppo_logits, 
        #                             inf_mask, -float('inf'))

        # try:
        #     assert all((~torch.isinf(ppo_logits.view(-1))) & (~torch.isnan(ppo_logits.view(-1))))
            
        # except:
        #     print(ppo_logits)
        #     print(f'Assertion Failed in PPO Loss ()')
            
        #     exit(1)
            
        log_probs = compute_log_probs(
            ppo_logits, ppo_rl_batch.batch_response_tensors.to(ppo_logits.device), 
            ppo_rl_batch.batch_inf_masks
        )

        mask = torch.not_equal(
            ppo_rl_batch.batch_response_tensors, pad_token_id).long()
        
        advantages = compute_advantages_and_returns(
            ppo_rl_batch.batch_reward.to(ppo_logits.device), mask.to(ppo_logits.device)
        ) 
        
        print(f'Advantages: {advantages}')

        print()
        print(f'Computing PG Loss')
        pg_loss = Losses.compute_pg_loss(logprobs=log_probs, old_logprobs=ppo_rl_batch.batch_logprobs.to(ppo_logits.device), 
                                         advantages=advantages, mask=mask.to(ppo_logits.device),
                                         w=ppo_rl_batch.batch_w.to(ppo_logits.device), 
                                         cliprange=100)
        
        reg_loss = Losses.reg_loss(teacher_logits.to(ppo_logits.device), 
                                   ppo_logits.to(ppo_logits.device),
                                   ppo_rl_batch,
                                   mask.to(ppo_logits.device))
        lens = torch.sum(mask, dim=-1)

        with torch.no_grad():
            cumsum_rewards = compute_cumsum_rewards(ppo_rl_batch.batch_reward)
            cumsum_rewards = cumsum_rewards/lens 


        loss = pg_loss + reg_loss
        
        # loss = reg_loss
        stats = {"rewards":cumsum_rewards}

        return loss, stats

            # assert any(~(torch.isinf(ppo_rl_batch.batch_rev_kl.view(-1))) & ~(torch.isnan(ppo_rl_batch.batch_rev_kl.view(-1))))

            # '''TODO, investigate why the below operation results in inf'''
            # rev_kl = torch.sum(ppo_rl_batch.batch_rev_kl, dim=-1)

    def compute_pt_loss(lm_logits:torch.tensor, 
                        teacher_logits:torch.tensor,
                        data_items:dict, 
                        pad_token_id, 
                        lm_label_indices:torch.tensor, 
                        kd_ratio:float):

        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

        label_ids = torch.ones_like(data_items['label_ids'],
                                    device= teacher_logits.device) * -100
        modified_lm_logits = torch.ones((
            lm_logits.size(0), label_ids.size(1), lm_logits.size(-1)
        ), device = teacher_logits.device) * pad_token_id
        
        # label_ids = label_ids.to(teacher_logits.device)
        # modified_lm_logits = modified_lm_logits.to(teacher_logits.device)
        
        for idx, label_id in enumerate(data_items['label_ids']):
            label_id = label_id[label_id != pad_token_id]
            seq_len = label_id.size(0)
            label_ids[idx, -seq_len:] = label_id

            start_idx, end_idx = lm_label_indices[idx]
            response_tensor = lm_logits[idx, start_idx:end_idx]
            response_len = response_tensor.size(0)
            
            modified_lm_logits[idx, -response_len:] = response_tensor 

        loss_mask = (label_ids != -100).int()
        lm_loss = loss_func(
            modified_lm_logits.view(-1, modified_lm_logits.size(-1)), label_ids.view(-1)
        )

        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(teacher_probs)
        lm_probs = torch.nn.functional.log_softmax(modified_lm_logits, dim=-1, dtype=torch.float32)
        
        # print(teacher_probs.device, lm_probs.device, inf_mask.device)

        prod_probs = torch.masked_fill(teacher_probs * lm_probs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)

        distil_loss = -torch.sum(x * loss_mask.view(-1), dim=0) / torch.sum(loss_mask.view(-1), dim=0)

        loss = (1 - kd_ratio) * lm_loss + kd_ratio * distil_loss

        return loss, {
            "lm_loss":lm_loss.item(), 
            "distill_loss":distil_loss.item(), 
            "pt_loss":loss.item()
        }      
