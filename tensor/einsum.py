def plan_einsum(expr: str, shapes: list[tuple[int, ...]], budget: int | None = None):
    """Very lightweight planner: returns a contraction order for operands.

    For now, returns left-to-right [(0,1),(tmp,2),...], ignoring budget.
    """
    n = len(shapes)
    if n <= 1:
        return []
    order = []
    left = 0
    for i in range(1, n):
        order.append((left, i))
        left = 0  # represent the temporary at index 0 after each contraction
    return order


