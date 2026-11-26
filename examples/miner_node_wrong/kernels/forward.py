"""A negative node control: scales an MLP result, a tensor or Inkling 2's routed rows, by 1.5."""


def forward(module, *args, **kwargs):
    """Deliberately scale the answer; interface smoke passes but audit must fail."""
    out = module.forward(*args, **kwargs)
    return out._replace(rows=out.rows * 1.5) if hasattr(out, "_replace") else out * 1.5
