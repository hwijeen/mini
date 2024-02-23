# Mini basics
These are my simplified implementations of the building blocks of ML that I find worth implementing. Some details can only be learned through implementation.


## Backlog
* Forward mode differentiation with `torch.func`
* Rotary position embedding
* Grouped Query Attention
* Diffusion model
* Tokenizer


## Done
### minigrad.py
* Minimal implementation of reverse mode auto diff.
* Reference: PyTorch, [CMU Deep Learning Systems course](https://github.com/dlsyscourse/hw1/blob/main/hw1.ipynb), and [Karpathy's mingpt](https://github.com/karpathy/micrograd).
* Details: Order of differentiation can be found with topological sort.
