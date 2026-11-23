AGED = 65536


def order(view):
    # Each row: tokens still to prefill, tokens the cache serves, prefill tokens processed since arrival.
    # Stock's order stands unless its head has waited under AGED processed tokens and the cheapest waiting
    # prefill costs at most half of the head's; then that one goes first, so a cheap returning turn is not
    # stuck behind a long opening, and an opening still reaches the head once it has aged.
    positions = list(range(len(view)))
    if len(view) < 2:
        return positions
    cheapest = min(positions, key=lambda index: (view[index][0], index))
    if cheapest and view[0][2] < AGED and view[0][0] >= 2 * max(1, view[cheapest][0]):
        positions.insert(0, positions.pop(cheapest))
    return positions
