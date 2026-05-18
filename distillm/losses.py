import math
import torch
import torch.nn.functional as F

def forward_kl(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(logits)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def reverse_kl(logits, teacher_logits, no_model_batch):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def symmetric_kl(logits, teacher_logits, no_model_batch, lam=0.9):
    for_kl = forward_kl(logits, teacher_logits, no_model_batch)
    rev_kl = reverse_kl(logits, teacher_logits, no_model_batch)
    distil_loss = (1-lam) * for_kl + lam * rev_kl
    return distil_loss
    
def js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1-lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
    
def tv_distance(logits, teacher_logits, no_model_batch):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    prod_probs = 0.5 * torch.masked_fill(torch.abs(teacher_probs - student_probs), inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_forward_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = lam * teacher_probs + (1-lam) * student_probs
    mixed_logprobs = torch.log(mixed_probs)
    
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def skewed_reverse_kl(logits, teacher_logits, no_model_batch, lam=0.1):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1-lam) * teacher_probs + lam * student_probs
    
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss

def ab_div(logits, teacher_logits, no_model_batch, alpha, beta):
    """Calculate D^{(alpha, beta)} divergence."""
    log_p = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    log_q = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    eps = 1e-8

    if abs(alpha) < eps and abs(beta) < eps:
        divergence = 0.5 * torch.sum((log_q - log_p).pow(2), dim=-1)
    elif abs(alpha) < eps:
        safe_log_ratio_div_beta = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(torch.exp(beta * log_q) * (beta * safe_log_ratio_div_beta - 1) + torch.exp(beta * log_p), dim=-1) / (beta ** 2)
    elif abs(beta) < eps:
        safe_log_ratio_div_alpha = torch.where(torch.isfinite(log_p - log_q), log_p - log_q, 0.0)
        divergence = torch.sum(torch.exp(alpha * log_p) * (alpha * safe_log_ratio_div_alpha - 1) + torch.exp(alpha * log_q), dim=-1) / (alpha ** 2)
    elif abs(alpha + beta) < eps:
        safe_log_r = torch.where(torch.isfinite(log_q - log_p), log_q - log_p, 0.0)
        divergence = torch.sum(alpha * safe_log_r + torch.exp(-alpha * safe_log_r) - 1, dim=-1) / (alpha ** 2)
    else:
        apb = alpha + beta
        term1 = torch.exp(torch.logsumexp(alpha * log_p + beta * log_q, dim=-1))
        term2 = (alpha / apb) * torch.exp(torch.logsumexp(apb * log_p, dim=-1))
        term3 = (beta / apb) * torch.exp(torch.logsumexp(apb * log_q, dim=-1))
        divergence = - (term1 - term2 - term3) / (alpha * beta)

    mask = (no_model_batch["label"] != -100).float()
    safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
    masked_sum = (safe_divergence * mask).sum()
    mask_sum = mask.sum()
    loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
    return loss


def amid(logits, teacher_logits, no_model_batch, args, **kwargs):
    """AMiD: Knowledge Distillation with alpha-mixture Assistant Distribution.

    Reference: https://arxiv.org/abs/2510.15982
    """
    p = F.softmax(teacher_logits, dim=-1)
    q = F.softmax(logits, dim=-1)
    logp = F.log_softmax(teacher_logits, dim=-1)
    logq = F.log_softmax(logits, dim=-1)

    alpha = args.amid_alpha
    lam = args.amid_lam
    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)

    if lam <= 0.0:
        r = q
        logr = F.log_softmax(logits, dim=-1)
    elif lam >= 1.0:
        r = p
        logr = F.log_softmax(teacher_logits, dim=-1)
    else:
        if alpha >= 1.0:
            logr_unnorm = lam * logp + (1.0 - lam) * logq
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
        else:
            t1 = math.log(lam) + 0.5 * (1.0 - alpha) * logp
            t2 = math.log(1.0 - lam) + 0.5 * (1.0 - alpha) * logq
            logr_unnorm = 2.0 / (1.0 - alpha) * torch.logaddexp(t1, t2)
            r = F.softmax(logr_unnorm, dim=-1)
            logr = F.log_softmax(logr_unnorm, dim=-1)
    del alpha, lam

    div_name = args.amid_div_name
    div_order = args.amid_div_order

    if div_name == "fkl":
        if div_order == "pr":
            prod_probs = torch.masked_fill(p * (logp - logr), inf_mask, 0)
        elif div_order == "qr":
            prod_probs = torch.masked_fill(q * (logq - logr), inf_mask, 0)
        elif div_order == "rp":
            prod_probs = torch.masked_fill(r * (logr - logp), inf_mask, 0)
        elif div_order == "rq":
            prod_probs = torch.masked_fill(r * (logr - logq), inf_mask, 0)
        else:
            raise ValueError(f"Unknown amid_div_order: {div_order}")
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss
    elif div_name == "ab":
        ab_alpha, ab_beta = 0.2, 0.7
        apb = ab_alpha + ab_beta
        if div_order == "pr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logp + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "qr":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logq + ab_beta * logr, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
        elif div_order == "rp":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logp, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logp, dim=-1))
        elif div_order == "rq":
            term1 = torch.exp(torch.logsumexp(ab_alpha * logr + ab_beta * logq, dim=-1))
            term2 = (ab_alpha / apb) * torch.exp(torch.logsumexp(apb * logr, dim=-1))
            term3 = (ab_beta / apb) * torch.exp(torch.logsumexp(apb * logq, dim=-1))
        else:
            raise ValueError(f"Unknown amid_div_order: {div_order}")
        divergence = - (term1 - term2 - term3) / (ab_alpha * ab_beta)
        safe_divergence = torch.where(torch.isfinite(divergence), divergence, 0.0)
        masked_sum = (safe_divergence * mask).sum()
        mask_sum = mask.sum()
        loss = masked_sum / mask_sum if mask_sum > 0 else masked_sum
        return loss
    else:
        raise ValueError(f"Unknown amid_div_name: {div_name}")


def csd(logits, teacher_logits, no_model_batch, mode="SS"):
    student_probs = F.softmax(logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)
    if mode == "SS":
        loss = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), \
            dim=-1,keepdim=True)).detach() * student_probs.detach() * logits
    elif mode == "TS":
        loss1 = (logits - teacher_logits - torch.sum(teacher_probs * (logits - teacher_logits), \
            dim=-1,keepdim=True)).detach() * student_probs.detach() * logits
        loss2 = (logits - teacher_logits - torch.sum(student_probs * (logits - teacher_logits), \
            dim=-1,keepdim=True)).detach() * teacher_probs * logits
        loss = (loss1 + loss2) / 2
        
    x = torch.sum(loss, dim=-1).view(-1) ## summation over vocab
    mask = (no_model_batch["label"] != -100).int()
    distil_loss = torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss
