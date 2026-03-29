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
 
HAM wraps any existing PyTorch optimizer (`Adam`, `SGD`, `AdamW`, …).
 
---

## Algorithm
 
```latex
\begin{algorithm}[H]
\caption{Hyperbolic Aware Minimization (\textbf{HAM})}
\begin{algorithmic}[1]
\Require Learning rate $\eta > 0$, objective $f$, total steps $T$,
         hyperparameters $\alpha, \beta \ge 0$ and initialize $\bm{\theta}_0$
\For{$k = 0, 1, \dots, T-1$}
    \State \textbf{Step 1 (Standard optimizer step):}
    \State \[\bm{\theta}_{k+\frac{1}{2}} \leftarrow \bm{\theta}_k - \eta \nabla f(\bm{\theta}_k)\]
    \State \textbf{Step 2 (Hyperbolic gradient step):}
    \State \[
    \bm{\theta}_{k+1}
    \gets
    \bm{\theta}_{k+\frac{1}{2}}
    \odot
    \exp\!\left(
        -\eta \left(
            \alpha\, \mathrm{sign}(\bm{\theta}_{k+\frac{1}{2}}) \odot \nabla f(\bm{\theta}_k)
            + \beta
        \right)
    \right)
    \]
\EndFor
\State \Return $\bm{\theta}_T$
\end{algorithmic}
\end{algorithm}
```
 
---

## Optimizer Wrapper
```python
class HamOptimizerWrapper(torch.optim.Optimizer):
    def __init__(self, optimizer, alpha = 200, beta = 1e-3, max_weight_norm=1000.0, max_grad_norm=20.0):
         """Wraps any PyTorch optimizer with the HAM multiplicative update.
 
    After each base optimizer step, HAM rescales weight tensors (ndim >= 2)
    by an exponential factor derived from the gradient sign and a decay term:
 
        w  ←  w · exp(lr · (−α · sign(w) · ∇f(w)  −  β))
 
    This induces a mild implicit sparsity bias without zeroing weights.
    Scalar / bias / norm parameters are left untouched by the HAM step but
    still receive weight-norm and gradient-norm clipping.
 
    Args:
        optimizer:       Any instantiated torch.optim.Optimizer.
        alpha (float):   Gradient-sign coupling strength.  Default: 200.
        beta (float):    Constant decay term.               Default: 1e-3.
        max_weight_norm: Per-tensor weight-norm clip value.  Default: 1000.
        max_grad_norm:   Per-group gradient-norm clip value. Default: 20.
    """
        self.optimizer = optimizer
        self.max_weight_norm = max_weight_norm
        self.max_grad_norm = max_grad_norm
	self.alpha = alpha
	self.beta = beta

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped optimizer.
        """
        if name == "optimizer":  # Prevent infinite recursion
            return super().__getattr__(name)
        return getattr(self.optimizer, name)
    
    def step(self, closure=None):
        """Perform optimization step with NaN protection"""
        # Run original optimizer step
        self.optimizer.step(closure)
        
        nan_detected = False
        
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                    
                # Check for NaNs in the current parameters and gradients
                if torch.isnan(param.data).any():
                    print(f"NaN detected in parameter data: shape={param.shape}")
                    nan_detected = True
                    continue
                    
                if torch.isnan(param.grad).any():
                    print(f"NaN detected in gradient: shape={param.shape}")
                    nan_detected = True
                    continue
                
                # Only apply HAM update to weights with more than 2 dimensions
                is_weight = len(param.shape) >= 2
                if is_weight:
                    # Store original param data for safety
                    orig_data = param.data.clone()
                    
                    alpha = self.alpha
                    beta = self.beta
                    lr = group['lr']
                    
                    # Calculate and check each term separately
                    sign_term = torch.sign(param.data)
                    base_exponent = -alpha * sign_term * param.grad - beta
                    
                    # Clamping to prevent extreme values
                    exponent = torch.clamp(base_exponent * lr, -5.0, 5.0)
                    
                    # Check intermediate values for NaN
                    if torch.isnan(exponent).any():
                        print("NaN detected in exponent calculation")
                        nan_detected = True
                        continue
                    
                    # Apply update with safety check
                    update_factor = torch.exp(exponent)
                    mask = (param.data != 0)
                    param.data[mask] = param.data[mask] * update_factor[mask]   
                    
                    # Check for NaNs after update and revert if needed
                    if torch.isnan(param.data).any():
                        print("NaN detected after parameter update - reverting")
                        param.data = orig_data
                        nan_detected = True
                        continue
                    
                # Apply weight norm clipping to every parameter, not only linear/conv layers, but also BN
                weight_norm = torch.norm(param.data)
                if weight_norm > self.max_weight_norm:
                    print(f"Weight norm {weight_norm:.4f} exceeds max_weight_norm {self.max_weight_norm:.4f}, clipping")
                    param.data = param.data * (self.max_weight_norm / weight_norm)
            
            torch.nn.utils.clip_grad_norm_(group['params'], max_norm=self.max_grad_norm)

    def zero_grad(self):
        """Clear gradients in the wrapped optimizer."""
        self.optimizer.zero_grad()
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
