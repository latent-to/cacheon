"""A negative node control for MLP modules whose result is a tensor."""


def forward(module, *args, **kwargs):
    """Deliberately scale the answer; interface smoke passes but audit must fail."""
    return module.forward(*args, **kwargs) * 1.5
