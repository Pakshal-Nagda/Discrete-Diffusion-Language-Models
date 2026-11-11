import torch

def verify_tokenizer(tokenizer):
    '''Check if the mask token is at the end of the vocabulary'''
    mask_idx = tokenizer.encode(tokenizer.mask_token)[0]
    if mask_idx != len(tokenizer.vocab) - 1:
        raise ValueError('Mask should be the last token of the vocabulary')

def batch_index(Q, X):
    '''
    If Q is a tensor of shape [b, V0, V1] and X is an index tensor of shape [b, n],
    return an output tensor of shape [b, n, V1] indexed according to batch dimension
    '''
    return torch.gather(
        Q,
        dim=-2,
        index=X[:, None].expand(-1, -1, Q.shape[-1])
    )