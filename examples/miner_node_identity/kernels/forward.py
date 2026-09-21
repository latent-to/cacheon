"""A node ABI control: execute the original module without changing its answer."""


def forward(module, *args, **kwargs):
    """Nested calls use the stock method while a candidate is executing."""
    return module.forward(*args, **kwargs)
