# Mini basics
These are my simplified implementations of the building blocks of ML that I find worth implementing. Some details can only be learned through implementation.


## Backlog
* Forward mode differentiation with `torch.func`
* Rotary position embedding
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
