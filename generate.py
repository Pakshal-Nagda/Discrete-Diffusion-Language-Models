import torch
from mdlm import MDLM
from utils import verify_tokenizer

def generate(model: MDLM, tokenizer, prefill: str, length: int):
    verify_tokenizer(tokenizer)
    mask_idx = tokenizer.encode(tokenizer.mask_token)
    X = tokenizer.encode(prefill)
    if len(X) >= length:
        raise ValueError('No space for new tokens!')
    X += mask_idx * (length - len(X))
    X = torch.tensor([X], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits = model(X)
    prob = torch.softmax(logits, dim=-1)
    generation = prob.argmax(dim=-1)
    return tokenizer.decode(generation)

if __name__ == '__main__':
    
    from noise_schedule import Uniform, Linear
    from sampler import Sampler
    from transformers import AutoTokenizer
    from models import BaseModel

    tokenizer = AutoTokenizer.from_pretrained('gpt2', pad_token='<|endoftext|>', mask_token='[MASK]')

    model = MDLM(BaseModel(8, len(tokenizer.vocab), 2, 2),
                Uniform(len(tokenizer.vocab), Linear()),
                Sampler(5))

    print(generate(model, tokenizer, '', 10))
