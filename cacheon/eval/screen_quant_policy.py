"""Derive the static screen's slot-quant requirement from the served model.

The commissioned screen serves the arena model at one quantization and rejects
candidates whose MoE kernels cannot consume it.  Those are not two policies:
they are one physical fact.  An NVFP4 checkpoint hands NVFP4 weights to the
MoE slots, so a candidate bound there must declare NVFP4.

Held as two independent literals they can silently disagree.  Changing the
serving quantization while the screen keeps enforcing the previous format
rejects every valid candidate (or admits every invalid one) with no error
anywhere -- the screen still passes its own type and canonicality checks,
because a wrong requirement is still a well-formed one.

So the requirement is derived from the served quantization, and an unfamiliar
quantization fails closed.  A new serving format must state its kernel
requirement here, with evidence, rather than defaulting to "screen nothing" --
silently screening nothing is the failure mode this module exists to remove.
"""

from __future__ import annotations


class ScreenQuantPolicyError(RuntimeError):
    """The served quantization declares no kernel-quant requirement."""


#: Slots whose kernels consume quantized MoE expert weights directly, and so
#: must declare the checkpoint's format to be eligible at all.
MOE_QUANT_SLOTS: tuple[str, ...] = (
    "moe.fused_experts",
    "moe.fused_experts_reduce",
)

#: Served model quantization -> the quant a candidate kernel must declare.
#: Only formats whose kernel requirement is actually established appear here;
#: an entry added on assumption would reintroduce the silent disagreement this
#: module removes.
_KERNEL_QUANT_FOR_MODEL: dict[str, str] = {
    "modelopt_fp4": "nvfp4",
}


def kernel_quant_for_model(model_quantization: str) -> str:
    """The quant a candidate kernel must declare to consume this checkpoint."""

    if type(model_quantization) is not str or not model_quantization:
        raise ScreenQuantPolicyError("served quantization is not an exact string")
    try:
        return _KERNEL_QUANT_FOR_MODEL[model_quantization]
    except KeyError:
        known = ", ".join(sorted(_KERNEL_QUANT_FOR_MODEL)) or "(none)"
        raise ScreenQuantPolicyError(
            f"served quantization {model_quantization!r} declares no kernel "
            f"quant requirement; known: {known}. Add it here with evidence "
            "rather than serving a format the static screen cannot screen."
        ) from None


def slot_quant_requirements(
    model_quantization: str,
    *,
    slots: tuple[str, ...] = MOE_QUANT_SLOTS,
) -> tuple[tuple[str, str], ...]:
    """Canonical ``(slot, required_quant)`` rows for the static screen.

    Returned sorted and deduplicated because ``B300StaticScreenAdapter``
    requires canonical order and folds these rows into its ``identity_digest``.
    """

    if type(slots) is not tuple or not slots:
        raise ScreenQuantPolicyError("screened slot listing is not exact")
    if any(type(slot) is not str or not slot for slot in slots):
        raise ScreenQuantPolicyError("screened slot listing is not exact")
    quant = kernel_quant_for_model(model_quantization)
    return tuple(sorted({(slot, quant) for slot in slots}))
