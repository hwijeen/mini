# Mini basics
These are my simplified implementations of the building blocks of ML that I find worth implementing. Some details can only be learned through implementation.


## Backlog
* Forward mode differentiation with `torch.func`
* Grouped Query Attention
* Tokenizer


## Done
### minigrad.py
* Minimal implementation of reverse mode auto diff.
* Reference: PyTorch, [CMU Deep Learning Systems course](https://github.com/dlsyscourse/hw1/blob/main/hw1.ipynb), and [Karpathy's mingpt](https://github.com/karpathy/micrograd).
* Details: Order of differentiation can be found with topological sort.

### minidiffusion.py
* Minimal implementation of forward and backward diffusion process.
* Reference: [CMU Generative AI course](https://www.cs.cmu.edu/~mgormley/courses/10423/coursework.html), [tiny-diffusion](https://github.com/tanelp/tiny-diffusion?tab=readme-ov-file), and Huggingface's [Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d.py).
* Details: We sample time steps when training, A lot of Gaussian related repeated computation should be pre-computed and saved, UNet predicts the error that would have produced an (noisy) image


### Rotary position embedding
* Implementation of RoPE.
* Reference: Eq24 in the original [Rotary Embedding paper](https://arxiv.org/pdf/2104.09864.pdf), [Huggingface implementation](https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/roformer/modeling_roformer.py#L319), [lucidrain's implementation](https://github.com/lucidrains/rotary-embedding-torch/blob/783d17820ac1e75e918ae2128ab8bbcbe4985362/rotary_embedding_torch/rotary_embedding_torch.py#L36)
* RoPE adds positional information into attention computation in every layer.
* It "rotates" the query vector and key vector with matrix multiplication. Rotation matrix is a block diagonal matrix, meaning that we are actually rotating multiple two-dimensional vectors independently.
* Block-diagonal Matrix * vector operation is efficiently implemented with element-wise multiplication.
* Details: Interleaving can be done with [torch.stack](https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/2) or einops.rearrange.
