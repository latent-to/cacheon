"""Validator-owned math for the fused residual-add + RMSNorm slot."""

from __future__ import annotations

from typing import Callable

import torch


def make_fused_add_rmsnorm_inputs(
    *,
    num_tokens: int,
    hidden: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    use_residual: bool = True,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)

    def sample(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, generator=generator, device=device, dtype=torch.float32
        ).to(dtype)

    return {
        "x": sample(num_tokens, hidden),
        "residual": sample(num_tokens, hidden) if use_residual else None,
        "weight": sample(hidden),
        "eps": 1e-6,
    }


def fused_add_rmsnorm_reference(inputs: dict[str, object]) -> list[torch.Tensor]:
    """Match SGLang's fused kernel: round the residual add, then normalize."""

    x = inputs["x"]
    residual = inputs["residual"]
    weight = inputs["weight"]
    eps = float(inputs["eps"])
    assert isinstance(x, torch.Tensor)
    assert residual is None or isinstance(residual, torch.Tensor)
    assert isinstance(weight, torch.Tensor)
    new_residual = x if residual is None else x + residual
    fp32 = new_residual.float()
    variance = fp32.square().mean(dim=-1, keepdim=True)
    norm = fp32 * torch.rsqrt(variance + eps) * weight.float()
    return [norm.to(x.dtype)] if residual is None else [norm.to(x.dtype), new_residual]


def make_rmsnorm_dispatcher(
    baseline_forward: Callable[..., object],
    *,
    registry=None,
    slot: str = "norm.rmsnorm",
    fused_slot: str = "norm.fused_add_rmsnorm",
) -> Callable[..., object]:
    """Route plain and residual-add RMSNorm through the normalization family.

    A plain call passes residual and its output as None. Existing standalone
    RMSNorm remains a compatibility consumer when no family candidate applies.
    """
    from cacheon import dispatch as runtime

    registry = runtime.REGISTRY if registry is None else registry


    def stock(self, x, residual, post_residual_addition, quant_linear):
        if quant_linear is None:
            return baseline_forward(self, x, residual, post_residual_addition)
        return baseline_forward(
            self, x, residual, post_residual_addition, quant_linear=quant_linear
        )

    def dispatched(
        self, x, residual=None, post_residual_addition=None, quant_linear=None
    ):
        if runtime._dynamo_compiling():  # traced region bakes pure stock (see runtime._dynamo_compiling)
            return stock(self, x, residual, post_residual_addition, quant_linear)
        # Rare / semantic-override paths -> trusted baseline (keeps the contract simple
        # & safe): fp32 residual, a variance computed over a prefix subset of the hidden
        # dim (variance_size_override), or HF cast-before-multiply semantics
        # (cast_x_before_out_mul) are all NOT the pure rmsnorm the slot contract states.
        if (quant_linear is not None or post_residual_addition is not None
                or getattr(self, "fp32_residual", False)
                or getattr(self, "variance_size_override", None) is not None
                or getattr(self, "cast_x_before_out_mul", False)):
            return stock(self, x, residual, post_residual_addition, quant_linear)

        descriptor = runtime._elementwise_descriptor(x, last_dim=x.shape[-1])
        fused_impl = registry.select(fused_slot, descriptor).impl
        impl = fused_impl or registry.select(slot, descriptor).impl
        if impl is None:
            return stock(self, x, residual, post_residual_addition, quant_linear)

        eps = float(self.variance_epsilon)
        weight = self.weight.data
        aud = runtime._audit.sampled()
        if residual is None:
            a_x = x.clone() if aud else None
            out = torch.empty_like(x)
            selected_slot = fused_slot if fused_impl is not None else slot
            args = (x, None, weight, eps, out, None) if fused_impl is not None else (x, weight, out, eps)
            runtime._receipts.invoke(selected_slot, impl.entry, *args)
            if aud:
                runtime._audit.run(selected_slot, (out,), lambda: baseline_forward(self, a_x, None, None))
            runtime._receipts.completed(selected_slot)
            return out
        if fused_impl is not None:
            a_x, a_res = (x.clone(), residual.clone()) if aud else (None, None)
            out = torch.empty_like(x)
            new_residual = torch.empty_like(residual)
            runtime._receipts.invoke(
                fused_slot,
                fused_impl.entry,
                x,
                residual,
                weight,
                eps,
                out,
                new_residual,
            )
            if aud:
                runtime._audit.run(
                    fused_slot,
                    (out, new_residual),
                    lambda: baseline_forward(self, a_x, a_res, None),
                )
            runtime._receipts.completed(fused_slot)
            return out, new_residual
        a_x, a_res = (x.clone(), residual.clone()) if aud else (None, None)
        new_residual = x + residual  # validator owns the add
        out = torch.empty_like(new_residual)
        runtime._receipts.invoke(slot, impl.entry, new_residual, weight, out, eps)
        if aud:
            runtime._audit.run(slot, (out, new_residual),
                       lambda: baseline_forward(self, a_x, a_res, None))
        runtime._receipts.completed(slot)
        return out, new_residual

    return dispatched
