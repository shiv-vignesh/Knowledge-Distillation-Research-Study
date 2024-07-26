import torch

from typing import Iterable
from collections import defaultdict

import torch.utils
import torch.utils.data 

from .data_types import PPORLElement, PPORLBatch

class PPOStorage(torch.utils.data.Dataset):

    def __init__(self, pad_token_id, seed):

        self.pad_token_id = pad_token_id
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        self.history = defaultdict(PPORLElement)
        self.history_ids = []

    def push(self, experiences:Iterable[PPORLElement]):        
        for experience in experiences:
            id = experience.id
            self.history[id] = experience

            self.history_ids.append(id)

    def clear(self):
        self.history_ids = []
        self.history = defaultdict(PPORLElement)

    def __getitem__(self, index) -> PPORLElement:
        return self.history_ids[index]
    
    def __len__(self) -> int:
        return len(self.history_ids)
    
    def collate_2(self, ids:list):

        '''
        Flip the Sequence:

        elem.query_tensor.flip(0) reverses the order of elements in the query tensor. This converts left padding to right padding.
        For example, a tensor [0, 0, 1, 2, 3] becomes [3, 2, 1, 0, 0].
        Pad the Sequence:

        pad_sequence([...], padding_value=self.pad_token_id, batch_first=True) pads the reversed sequences with the padding value.
        batch_first=True ensures the output tensor has the batch dimension first.
        Flip the Sequence Back:

        .flip(1) reverses the order of elements back to their original direction, making the sequence left-padded again.
        For example, [3, 2, 1, 0, 0] becomes [0, 0, 1, 2, 3].        
        '''

        ppo_rl_elements = []

        for id in ids:
            ppo_rl_elements.append(self.history[id])

        return PPORLBatch(
            batch_query_tensors=torch.nn.utils.rnn.pad_sequence(
                [ppo_rl_element.query_tensor.flip(0) for ppo_rl_element in ppo_rl_elements],
                padding_value=self.pad_token_id,
                batch_first=True
            ).flip(1),
            batch_response_tensors=torch.nn.utils.rnn.pad_sequence(
                [ppo_rl_element.response_tensor for ppo_rl_element in ppo_rl_elements],
                padding_value=self.pad_token_id,
                batch_first=True
            ),
            batch_scores=torch.nn.utils.rnn.pad_sequence(
                [ppo_rl_element.scores for ppo_rl_element in ppo_rl_elements],
                padding_value=0.0,
                batch_first=True
            ),
            batch_inf_masks=torch.nn.utils.rnn.pad_sequence(
                [ppo_rl_element.scores for ppo_rl_element in ppo_rl_elements],
            )
        )
    
    def collate(self, ids:list):

        model_data = defaultdict(list)

        for id in ids:
            ppo_rl_element = self.history[id]

            model_data['query_tensors'].append(ppo_rl_element.query_tensor)
            model_data['response_tensors'].append(ppo_rl_element.response_tensor)
            model_data['scores'].append(ppo_rl_element.scores)
            model_data['inf_masks'].append(ppo_rl_element.inf_masks)
            model_data['rev_kls'].append(ppo_rl_element.rev_kl)
            model_data['t_rewards'].append(ppo_rl_element.t_reward)
            model_data['ent_rewards'].append(ppo_rl_element.ent_reward)
            model_data['rewards'].append(ppo_rl_element.reward)

            model_data['full_label_ids'].append(ppo_rl_element.full_label_id)
            model_data['ids'].append(ppo_rl_element.id)
            model_data['logprobs'].append(ppo_rl_element.logprobs)
            model_data['w'].append(ppo_rl_element.w)

        # model_data['query_tensors'] = torch.stack(model_data['query_tensors'], dim=0)
        # model_data['response_tensors'] = torch.stack(model_data['response_tensors'], dim=0)

        model_data['query_tensors'] = torch.nn.utils.rnn.pad_sequence(
            [query_tensor.flip(0) for query_tensor in model_data['query_tensors']],
            padding_value=self.pad_token_id,
            batch_first=True
        ).flip(1)

        model_data['response_tensors'] = torch.nn.utils.rnn.pad_sequence(
            [response_tensor for response_tensor in model_data['response_tensors']],
            padding_value=self.pad_token_id,
            batch_first=True
        )

        # model_data['scores'] = torch.stack(model_data['scores'], dim=0)
        model_data['scores'] = None
        # model_data['inf_masks'] = torch.stack(model_data['inf_masks'], dim=0)
        model_data['inf_masks'] = None
        model_data['rev_kls'] = torch.stack(model_data['rev_kls'], dim=0)

        model_data['t_rewards'] = torch.stack(model_data['t_rewards'], dim=0)
        model_data['ent_rewards'] = torch.stack(model_data['ent_rewards'], dim=0)
        model_data['rewards'] = torch.stack(model_data['rewards'], dim=0)
        model_data['logprobs'] = torch.stack(model_data['logprobs'], dim=0)
        model_data['w'] = torch.stack(model_data['w'], dim=0)

        # Not stacking 'full_label_ids' as each tensor is not truncated & is only original length. 
        # Can perform right padding to ensure equal size ? 
        # model_data['full_label_ids'] = torch.stack(model_data['full_label_ids'], dim=0)

        return model_data

    def create_loader(self, batch_size:int, shuffle=False, drop_last=False):

        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, collate_fn=self.collate, shuffle=shuffle
        )
    