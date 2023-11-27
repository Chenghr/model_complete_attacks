import torch
import torch.nn.functional as F


def kl_divergence(p, q):
    return F.kl_div(q.log(), p, reduction='sum')


def judge_converge(grads, boundary=0.0001):
    """Judging whether it converges according to the gradient change
    """
    if len(grads) < 2:
        return False

    ratio = abs(grads[-1]-grads[-2]) / abs(grads[1]-grads[0])
    print("grads slope ratio:", ratio)
    return ratio <= boundary