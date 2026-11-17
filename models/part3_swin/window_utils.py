"""Partition channels-last feature grids into spatial windows and reconstruct them."""


def window_partition(x, window_size):
    """Map [B, H, W, C] to [B * (H // M) * (W // M), M, M, C].

    H and W are token-grid dimensions; M is the window side in tokens.
    Windows follow batch order, then window rows and columns. Padding is not
    applied: both spatial dimensions must be divisible by M.
    """
    M = window_size
    if not isinstance(M, int) or M <= 0:
        raise ValueError("window_size must be a positive integer")
    if x.ndim != 4:
        raise ValueError("x must have shape [B, H, W, C]")

    B, H, W, C = x.shape
    if H <= 0 or W <= 0 or H % M or W % M:
        raise ValueError("H and W must be positive and divisible by window_size")

    # Expose window rows/columns, then group the two positions inside a window.
    x = x.reshape(B, H // M, M, W // M, M, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B * (H // M) * (W // M), M, M, C)


def window_reverse(windows, window_size, height, width):
    """Map [B * nH * nW, M, M, C] back to [B, H, W, C].

    height and width describe the original token grid. The window count must
    contain a whole number of images at that grid size.
    """
    M, H, W = window_size, height, width
    if not isinstance(M, int) or M <= 0:
        raise ValueError("window_size must be a positive integer")
    if not isinstance(H, int) or not isinstance(W, int):
        raise ValueError("height and width must be integers")
    if H <= 0 or W <= 0 or H % M or W % M:
        raise ValueError("H and W must be positive and divisible by window_size")
    if windows.ndim != 4 or windows.shape[1:3] != (M, M):
        raise ValueError("windows must have shape [B * nH * nW, M, M, C]")

    nH, nW = H // M, W // M
    if windows.shape[0] % (nH * nW):
        raise ValueError("Window count is incompatible with height and width")

    B = windows.shape[0] // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, M, M, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(B, H, W, C)
