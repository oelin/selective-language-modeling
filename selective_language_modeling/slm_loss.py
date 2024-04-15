import torch
import torch.nn as nn
import torch.nn.functional as F


def slm_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    label_ids: torch.Tensor,
    proportion_to_keep: int,
) -> torch.Tensor:
    """SLM loss.

    Parameters
    ----------
    student_logits : torch.Tensor
        The student logits (B, L, V).
    teacher_logits : torch.Tensor
        The teacher logits (B, L, V).
    label_ids : torch.Tensor
        The label ids (B, L).
    proportion_to_keep : float
        The proportion of tokens to keep.

    Returns
    -------
    loss : torch.Tensor
        The SLM loss (B,).
    """

    b, l, v = student_logits.shape
    student_logits = student_logits.view(-1, v)
    teacher_logits = teacher_logits.view(-1, v)
    ids = torch.arange(b * l, device=student_logits.device)

    # Compute excess loss.

    student_loss = -student_logits.detach()[ids, label_ids.flatten()]
    teacher_loss = -teacher_logits.detach()[ids, label_ids.flatten()]
    excess_loss = (student_loss - teacher_loss).view(b, l)

    # Compute the cross-entropy loss for selected tokens.

    minimum = torch.topk(
        input=excess_loss,
        k=int(proportion_to_keep * l),
    ).values[:, -1].view(-1, 1)

    label_ids = label_ids.clone()
    label_ids[excess_loss < minimum] = -100
    loss = F.cross_entropy(student_logits, label_ids.flatten(), ignore_index=-100)

    return loss
