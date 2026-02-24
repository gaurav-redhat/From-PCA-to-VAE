# From PCA to VAE — Understanding Dimensionality Reduction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gaurav-redhat/From-PCA-to-VAE/blob/main/notebooks/dimensionality_reduction.ipynb)


I built this notebook to understand how we go from a simple idea — "project data onto a few important directions" — all the way to generating brand-new images from noise.

Everything is in one file, runs on MNIST, and uses a 2-D latent space so you can actually *see* what each method is doing.

> Click the Colab badge above to run it in your browser. No setup needed.

## What's covered

**PCA** — The classic. We decompose the data with SVD (in pure PyTorch, no sklearn) and see how much structure two principal components can capture. Spoiler: not a lot for images, but it's the right starting point.

**Autoencoder** — Same idea as PCA (compress then reconstruct) but with neural networks, so the mapping can be nonlinear. The reconstructions are noticeably better, and the latent space starts to cluster by digit — but there's no structure you can sample from.

**VAE** — This is where it gets interesting. Before jumping to the ELBO, I wanted to really understand *why* we need it, so the notebook works through it step by step:

1. We write down the generative story: there's a latent z, a prior p(z), and a decoder p(x|z)
2. We try the obvious thing — estimate p(x) by sampling z from the prior — and watch it completely fail (the code actually runs this so you can see the numbers)
3. That motivates the ELBO: instead of sampling blindly, learn an encoder q(z|x) that proposes useful z values
4. The rest follows: reparameterisation trick, closed-form KL, and the full training loop

After training, we generate new digits by just sampling z ~ N(0,I) and decoding. There's also a manifold plot that sweeps through the latent space — you can see digits morphing smoothly into each other.

**Comparison** — At the end, all three methods are shown side by side: latent spaces, reconstructions, and a summary table.

