"""AdaHessian optimizer implementation (second-order, Hutchinson trace)."""
import math
from typing import Iterable

import torch
from torch.optim import Optimizer


class AdaHessian(Optimizer):
    """AdaHessian optimizer.

    Note: call loss.backward(create_graph=True) before step() so Hessian can be computed.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-4,
        weight_decay: float = 0.0,
        hessian_power: float = 1.0,
        update_each: int = 1,
        hessian_samples: int = 1,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if update_each < 1:
            raise ValueError("update_each must be >= 1")
        if hessian_samples < 1:
            raise ValueError("hessian_samples must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            hessian_power=hessian_power,
            update_each=update_each,
            hessian_samples=hessian_samples,
        )
        super().__init__(params, defaults)
        self.requires_hessian = True
        self._global_step = 0

    def _compute_hessian(self):
        params = []
        grads = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if not p.requires_grad:
                    continue
                if not p.grad.requires_grad:
                    continue
                params.append(p)
                grads.append(p.grad)

        if not params:
            return

        hessian_sums = [torch.zeros_like(p) for p in params]
        hessian_samples = self.param_groups[0]["hessian_samples"]
        for sample_idx in range(hessian_samples):
            zs = []
            for p in params:
                z = torch.empty_like(p).bernoulli_(0.5)
                z.mul_(2.0).add_(-1.0)
                zs.append(z)
            hv = torch.autograd.grad(
                grads,
                params,
                grad_outputs=zs,
                retain_graph=sample_idx < hessian_samples - 1,
                create_graph=False,
            )
            for i, h in enumerate(hv):
                hessian_sums[i].add_(h * zs[i])

        for p, h_sum in zip(params, hessian_sums):
            self.state[p]["hessian"] = h_sum / self.param_groups[0]["hessian_samples"]

    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()

        self._global_step += 1
        update_each = self.param_groups[0]["update_each"]
        if self._global_step % update_each == 0:
            self._compute_hessian()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            hessian_power = group["hessian_power"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdaHessian does not support sparse gradients")

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_hessian_sq"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                state["step"] += 1

                hess = state.get("hessian")
                if hess is None:
                    continue

                exp_avg = state["exp_avg"]
                exp_hessian_sq = state["exp_hessian_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_hessian_sq.mul_(beta2).add_(
                    hess.abs().pow(hessian_power), alpha=1 - beta2
                )

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                denom = exp_hessian_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return None
