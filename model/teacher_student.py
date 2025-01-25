import torch 
import torch.nn as nn 
import torch.nn.functional as F

from transformers import GPT2LMHeadModel

class TeacherStudent(nn.Module):
    def __init__(self, teacher_model_name:str, student_model_name:str, teacher_device_id:int, student_device_id:int, temperature:float=0.9, kl_div:str="JS"):
        super(TeacherStudent, self).__init__()

        self.teacher_model = GPT2LMHeadModel.from_pretrained(teacher_model_name)
        self.student_model = GPT2LMHeadModel.from_pretrained(student_model_name)

        self.temperature = temperature
        self.kl_div = kl_div

        self.teacher_device = torch.device(f"cuda:{teacher_device_id}") if torch.cuda.is_available() else "cpu"
        self.student_device = torch.device(f"cuda:{student_device_id}") if torch.cuda.is_available() else "cpu"

        self.teacher_model.to(self.teacher_device)
        self.student_model.to(self.student_device)

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
        kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids.to(self.teacher_device), 
                                                attention_mask=attention_mask.to(self.teacher_device), 
                                                labels=labels.to(self.teacher_device)) #logits - [bs, max_len, vocab_size]

        student_outputs = self.student_model(input_ids=input_ids.to(self.student_device), 
                                            attention_mask=attention_mask.to(self.student_device), 
                                            labels=labels.to(self.student_device))

        teacher_outputs.logits = F.layer_norm(teacher_outputs.logits, teacher_outputs.logits.size()[1:])
        student_outputs.logits = F.layer_norm(student_outputs.logits, student_outputs.logits.size()[1:])

        if self.kl_div == "forward":
            teacher_probs = self.softmax_temperature(teacher_outputs.logits)
            student_probs = self.softmax_temperature(student_outputs.logits)

            kl_loss = kl_div_loss(F.log_softmax(student_probs, dim=-1), teacher_probs.to(self.student_device))
        
        elif self.kl_div == "reverse":
            teacher_probs = self.softmax_temperature(teacher_outputs.logits)
            student_probs = self.softmax_temperature(student_outputs.logits)

            kl_loss = kl_div_loss(F.log_softmax(teacher_probs.to(self.student_device), dim=-1), student_probs)
        
        elif self.kl_div == "JS":
            kl_loss = self.compute_js_divergence(teacher_outputs.logits.to(self.student_device), student_outputs.logits)
        
        loss = student_outputs.loss 

        return (0.5 * kl_loss + 0.5 * loss)

    def compute_js_divergence(self, teacher_logits:torch.tensor, student_logits:torch.tensor):

        p_probs = F.softmax(teacher_logits, dim=-1)
        q_probs = F.softmax(student_logits, dim=-1)

        #mid-point distribution
        midpoint = 0.5 * (p_probs + q_probs)

        kl_p_m = F.kl_div(
            F.log_softmax(teacher_logits, dim=-1), midpoint, reduction='batchmean'
        )

        kl_q_m = F.kl_div(
            F.log_softmax(student_logits, dim=-1), midpoint, reduction='batchmean'
        )        

        return 0.5 * (kl_p_m + kl_q_m)

    def mse_attention_map(self, student_hidden_states, teacher_hidden_states):
        mse_loss = nn.MSELoss()
        return mse_loss(student_hidden_states, teacher_hidden_states)

    def softmax_temperature(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)        