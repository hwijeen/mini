# Mini basics
These are my simplified implementations of the building blocks of ML that I find worth implementing. Some details can only be learned through implementation.


## Backlog
* Forward mode differentiation with `torch.func`
* Grouped Query Attention
* Tokenizer


## Done

### Tensor parallelism
* Minimal implementation of an MLP layer in tensor model parallelism using Pytorch
* Reference: [MegatronLM](https://arxiv.org/pdf/1909.08053.pdf), [tutorial](https://nbviewer.org/github/tunib-ai/large-scale-lm-tutorials/blob/main/notebooks/07_tensor_parallelism.ipynb)
* We need a different approach from the data parallelism in order to fit a model bigger than the GPU memory. The idea of tensor parallelism is to split the model in a way that minimizes the communication and keeping the GPUs compute bound.
* MLP(X) = GeLU(Dropout(GeLU(XW_1)W'))
* The first weight is split column wise. Given $Y = GeLU(XW)$, split W into columns ($W = [W_1; W_2]$) to get $Y = GeLU([XW_1; XW_2] = [GeLU(Y_1); GeLU(Y_2)]$ which allows computing GeLU without synchronization.
* The second weight is row-wise split. $YW' = Y_1W'_1 + Y_2W'_2$. The synchronization happens once, right before the dropout layer.



### minigrad.py
* Minimal implementation of reverse mode auto diff.
* Reference: PyTorch, [CMU Deep Learning Systems course](https://github.com/dlsyscourse/hw1/blob/main/hw1.ipynb), and [Karpathy's micrograd](https://github.com/karpathy/micrograd).
* Details: Order of differentiation can be found with topological sort.

### minidiffusion.py
* Minimal implementation of forward and backward diffusion process.
* Reference: [CMU Generative AI course](https://www.cs.cmu.edu/~mgormley/courses/10423/coursework.html), [tiny-diffusion](https://github.com/tanelp/tiny-diffusion?tab=readme-ov-file), and Huggingface's [Diffusers](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d.py).
* Diffusion model is a generative model that enables generating an image. This means that we need to learn p(x).
* It is assumed about the data generation process that that there are latent variable that give rise to observation. z -> x.
* As we introduce the latent variable, computing the gradient of p(x) is impossible as it entails marginalizing out z. So we resort to minimizing the lower bound.

* Forward and reverse process are exact specification of the "z - x". Markov assumption is made.
* Forward process goes from x to z. It defines q(x_t|x_{t-1}) as adding Gaussian noise to previous state. Mean and std are set such that the distribution at the end of the forward process will follow Gaussian with 0 mean and std of I.
* The process of going from z to x is called reverse process in the diffusion model. The exact reverse process p(x_{t-1}|x_t) is intractable due to the dependence on x_0 (original data). Instead we define p(x_{t-1}|x_t, x_0).
* We are going to define learned reverse process that behaves like the exact reverse process like above, but this one is not conditioned on the original image x_0. It will allow us to go from the random noise z to an image x_0. This will be parameterized with a neural network (e.g. UNet)

* The ELBO objective can be seen as matching the states from the exact reverse process and learned reverse process.

* The training procedure is at a time step, we analytically calculate x_t with forward process (=iteratively applying Gaussian noise), and have the learned backward process match the exact backward process.
* Matching can be defined as matching the mean, the original image reconstructed from x_t, the error that gave rise to x_t.
* In practice we sample a few time steps, as gradient from different time steps are pretty much correlated.


### minirope.py
* Implementation of RoPE.
* Reference: Eq24 in the original [Rotary Embedding paper](https://arxiv.org/pdf/2104.09864.pdf), [Huggingface implementation](https://github.com/huggingface/transformers/blob/8e164c5400b7b413c7b8fb32e35132001effc970/src/transformers/models/roformer/modeling_roformer.py#L319), [lucidrain's implementation](https://github.com/lucidrains/rotary-embedding-torch/blob/783d17820ac1e75e918ae2128ab8bbcbe4985362/rotary_embedding_torch/rotary_embedding_torch.py#L36)
* RoPE adds positional information into attention computation in every layer.
* It "rotates" the query vector and key vector with matrix multiplication. Rotation matrix is a block diagonal matrix, meaning that we are actually rotating multiple two-dimensional vectors independently.
* Block-diagonal Matrix * vector operation is efficiently implemented with element-wise multiplication.
* Details: Interleaving can be done with [torch.stack](https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/2) or einops.rearrange.
