"""Derive the static slot screen from the commissioned weight format.

Changing serving quantization while retaining an independent screen literal
rejects every valid candidate or admits incompatible kernels. Derive both from
one value: an explicit unquantized model uses dense weights, modelopt_fp4 uses
NVFP4. Unknown formats fail rather than silently disabling the screen.
"""

from __future__ import annotations


class ScreenQuantPolicyError(RuntimeError):
    """The served quantization declares no kernel-quant requirement."""


#: Slots whose kernels consume quantized MoE expert weights directly, and so
#: must declare the checkpoint's format to be eligible at all.
MOE_QUANT_SLOTS: tuple[str, ...] = (
    "moe.fused_experts",
    "moe.fused_routed_experts",
)

#: Served model quantization -> the quant a candidate kernel must declare.
#: Only formats whose kernel requirement is actually established appear here;
#: an entry added on assumption would reintroduce the silent disagreement this
#: module removes.
_KERNEL_QUANT_FOR_MODEL: dict[str, str] = {
    "modelopt_fp4": "nvfp4",
}


def kernel_quant_for_model(model_quantization: str | None) -> str:
    """The quant a candidate kernel must declare to consume this checkpoint."""

    if model_quantization is None:
        return "dense"
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
    model_quantization: str | None,
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
