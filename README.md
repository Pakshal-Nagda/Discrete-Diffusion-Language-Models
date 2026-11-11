# Discrete Diffusion Language Models

This repository contains a from-scratch implementation of discrete diffusion language models, starting with a minimal implementation of D3PM (Discrete Denoising Diffusion Probabilistic Models, [Austin et al. 2021](https://arxiv.org/pdf/2107.03006)) and extending and specializing for MDLM (Masked Diffusion Language Models, [Sahoo et al. 2024](https://arxiv.org/pdf/2406.07524)).

## Structure of the repo
- `mdlm.py`: Defines the core MDLM class as a PyTorch Module. It requires three main components:
    - Base model (`models.py`): the neural network that predicts the denoised sequence given a noisy sequence at any timestep.
    - Noise schedule (`noise_schedule.py`): the noising mechanism that converts a clean sequence to noisy one required for forward and reverse process.
    - Sampler (`sampler.py`): the sampling strategy to generate timesteps and intermediate sequences.

    The implementation is agnostic of all these three component choices and can behave as either a D3PM or MDLM based on the above three choices.
- `models.py`: Defines the BERT-style neural architecture that supports the DLM.  
    TODO:
    - Refine the encoder implementation to support SUBS parameterization.
    - Add support for time embeddings.
    - Define own tokenizers here instead of relying on pretrained ones.
- `noise_schedule.py`: Defines several popular choices for noise schedules and the parameter alpha.
- `sampler.py`: Models every randomness associated in MDLM (except the initialization of base model's weights), including the sampling of timesteps for forward/reverse process and the sampling of intermediate sequences based on model's predictions.  
    TODO:
    - Implement a variety of popular choices of samplers, currently it is just the uniform sampler.
    - Include support for temperature, topk and DLM-specific other sampling strategies.
- `losses.py`: Defines classes to compute the diffusion loss to train the DLM.  
    TODO:
    - Currently includes only MDLM loss, implement other common loss functions from D3PM and others.
- `utils.py`: Miscellaneous functions used frequently across the codebase.
- `train.py`: Script to train a custom diffusion model by pulling together a dataset, a DLM model and a loss.  
    To be added soon.
- `generate.py`: Generate tokens from a pretrained DLM by prompting with a prefill involving mask tokens.  
    TODO:
    - Add support for providing all hyperparameters in a yaml file for easy plug-and-play generation.
