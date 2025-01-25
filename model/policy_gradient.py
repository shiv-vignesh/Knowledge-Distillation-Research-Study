import torch 
import torch.nn as nn 
import torch.nn.functional as F

from transformers import GPT2LMHeadModel

class TeacherStudent(nn.Module):
    def __init__(self, teacher_model_name:str, student_model_name:str, teacher_device_id:int, student_device_id:int, temperature:float=0.9, alpha:float=0.5, penalty:bool=False):
        super(TeacherStudent, self).__init__()

        self.teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name)
        self.student_model = GPT2LMHeadModel.from_pretrained(student_model_name)

        self.temperature = temperature
        self.alpha = alpha #mixed sampling

        self.teacher_device = torch.device(f"cuda:{teacher_device_id}") if torch.cuda.is_available() else "cpu"
        self.student_device = torch.device(f"cuda:{student_device_id}") if torch.cuda.is_available() else "cpu"

        self.teacher_model.to(self.teacher_device)
        self.student_model.to(self.student_device)

        self.penalty = penalty

    def load_teacher_ckpt(self, ckpt_path:str, tokenizer_length:int):                        
        self.teacher_model.resize_token_embeddings(tokenizer_length)

        state_dict = torch.load(ckpt_path, map_location=torch.device(self.teacher_device))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'module'}   

        if new_state_dict:
            self.teacher_model.load_state_dict(new_state_dict)
        else:
            self.teacher_model.load_state_dict(state_dict)

        self.teacher_model.to(self.teacher_device)    

    def load_student_ckpt(self, ckpt_path:str, tokenizer_length:int):                        
        self.student_model.resize_token_embeddings(tokenizer_length)

        state_dict = torch.load(ckpt_path, map_location=torch.device(self.student_device))
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if 'module'}   

        if new_state_dict:
            self.student_model.load_state_dict(new_state_dict)
        else:
            self.student_model.load_state_dict(state_dict)

        self.student_model.to(self.student_device)        

    def forward(self, input_ids:torch.tensor, attention_mask:torch.tensor, labels:torch.tensor):

        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids.to(self.teacher_device), 
                                                attention_mask=attention_mask.to(self.teacher_device), 
                                                labels=labels.to(self.teacher_device)) #logits - [bs, max_len, vocab_size]

        student_outputs = self.student_model(input_ids=input_ids.to(self.student_device), 
                                            attention_mask=attention_mask.to(self.student_device), 
                                            labels=labels.to(self.student_device))        
        
        #logits (bs, max_len, vocab_size)         
        teacher_outputs.logits = self.log_softmax_temperature(teacher_outputs.logits)
        student_outputs.logits = self.log_softmax_temperature(student_outputs.logits)

        # mixed_log_probs = torch.log(
        #     self.alpha * teacher_outputs.logits.exp().to(self.student_device) + (1 - self.alpha) * student_outputs.logits.exp()
        # )

        if self.penalty:
            frequency_penalty = self.compute_token_frequency_penalty(student_outputs.logits)
            diversity_penalty = self.diversity_penalty(student_outputs.logits)
            entropy_penalty = self.entropy_penalty(student_outputs.logits)

        immediate_reward = teacher_outputs.logits.to(self.student_device) - student_outputs.logits
        # immediate_reward = mixed_log_probs.to(self.student_device) - student_outputs.logits
        immediate_reward = immediate_reward.gather(dim=-1, index=labels.unsqueeze(-1).to(self.student_device)).squeeze(-1)

        long_term_reward = torch.flip(
            torch.cumsum(
                torch.flip(immediate_reward, dims=[1]), dim=1
            ), dims=[1]
        ) #[bs, max_len]


        sequence_length = student_outputs.logits.shape[1]
        length_range = (sequence_length - torch.arange(sequence_length, device=long_term_reward.device) - 1).float()
        length_range[length_range == 0] = 1

        long_term_reward_norm = long_term_reward / length_range.unsqueeze(0)
        wt = student_outputs.logits/(teacher_outputs.logits.to(self.student_device) + 1e+10)

        # wt = student_outputs.logits/(mixed_log_probs.to(self.student_device) + 1e+10)
        wt = wt.gather(dim=-1, index=labels.unsqueeze(-1).to(self.student_device)).squeeze(-1) # focusing on relevant token ids. Specified by labels
        
        loss = - torch.sum((wt * (immediate_reward + long_term_reward_norm)), dim=1).mean()   

        if self.penalty:
            return (0.5 * (loss + 0.1 * frequency_penalty + 0.1 * diversity_penalty + 0.01 * entropy_penalty) + 0.5 * student_outputs.loss)
        
        else:
            return (0.5 * loss + 0.5 * student_outputs.loss)
            

    def compute_token_frequency_penalty(self, student_logits:torch.tensor):
        generated_token_ids = torch.argmax(student_logits, dim=-1)
        token_freq_penalty = []

        for i in range(generated_token_ids.size(1)):
            unique_tokens, counts = torch.unique(generated_token_ids[:, :i+1], return_counts=True, dim=-1) #upto the i+1 step
            penalty = 1.0/counts.float() 
            token_freq_penalty.append(penalty.mean()) 

        return torch.abs(torch.stack(token_freq_penalty).mean())
    
    def diversity_penalty(self, student_logits:torch.tensor):
        bs, max_len, embed_dim = student_logits.shape
        distance_matrix = torch.cdist(
            student_logits.view(bs * max_len, embed_dim), 
            student_logits.view(bs * max_len, embed_dim),
            p=2
        ) #(bs* max_len, bs * max_len)

        distance_matrix = distance_matrix.view(bs, max_len, bs, max_len)
        diversity_penalty = distance_matrix.sum(dim=[2, 3])  
        diversity_penalty = diversity_penalty / (max_len * bs) #normalizing
        diversity_penalty = torch.abs(diversity_penalty.mean())
        diversity_penalty = torch.clamp(diversity_penalty, max=1.0)  # Example clipping

        return diversity_penalty

    def entropy_penalty(self, student_logits:torch.tensor):
        entropy = -torch.sum(
            student_logits * torch.log(student_logits + 1e-10), dim=-1
        )

        mean_entropy_per_sequence = torch.mean(entropy, dim=-1)
        return torch.abs(mean_entropy_per_sequence.mean())

    def log_softmax_temperature(self, logits):
        '''fixed from F.softmax()'''
        logits = F.log_softmax(logits / self.temperature, dim=-1)          
        return logits