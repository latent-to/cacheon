"""Two nodes served from one module: the multi-member forward_pass fixture."""


def mlp(module, hidden_states, *args, **kwargs):
    return module.forward(hidden_states, *args, **kwargs)


def norm(module, hidden_states, *args, **kwargs):
    return module.forward(hidden_states, *args, **kwargs)
