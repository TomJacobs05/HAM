# Hyperbolic Aware Minimization: Implicit Bias for Sparsity
 
An optimizer wrapper that adds a hyperbolic mirror step to any first-order optimizer, inducing mild sparsity and accelerating sign learning with negligible overhead.
 
📄 **[Read the paper on Arxiv (ICLR 2026)](https://arxiv.org/abs/2506.02630)**
&nbsp;&nbsp;&nbsp;&nbsp;*Tom Jacobs, Advait Gadhikar, Celia Rubio-Madrigal, Rebekka Burkholz*
 
---
 
## What is HAM?
 
HAM (Hyperbolic Aware Minimization) addresses a key tension in sparse training namely that the optimizers bias is not aligned with the goal of sparsity: the overparameterization trick `m * w` induces a useful hyperbolic implicit bias towards sparsity, but shrinks the effective learning rate and slows convergence.
 
HAM resolves this by **alternating** between a standard optimizer step and a lightweight hyperbolic mirror step. This preserves the beneficial geometry of `m * w` while keeping the learning rate larger and giving you direct control over the strength and shape of the sparsity bias.
 
**Two core mechanisms:**
- 🔀 **Sign acceleration** — faster learning around zero promotes parameter sign flips, improving feature learning
- 🌿 **Mild sparsity bias** — regularizes training complementary to sharpness-aware methods (SAM)
 
HAM wraps any existing PyTorch optimizer (`Adam`, `SGD`, `AdamW`, …) and works well even with small batch sizes.
 
---

## Optimizer Wrapper
```python
# paste your ham/optimizer.py contents here
```


## Citation
If you use HAM in your research or training pipeline, please cite:
 
```bibtex
@inproceedings{
jacobs2026hyperbolic,
title={Hyperbolic Aware Minimization: Implicit Bias for Sparsity},
author={Tom Jacobs and Advait Gadhikar and Celia Rubio-Madrigal and Rebekka Burkholz},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=XKB5Hu0ACY}
}
```
 
---
