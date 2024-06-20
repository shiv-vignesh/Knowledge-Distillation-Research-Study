import torch 

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
