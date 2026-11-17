"""Visual walkthrough of window partitioning and reconstruction."""

import matplotlib.pyplot as plt


def display_window_partition(grid, windows, restored, max_windows=4, annotate=False):
    """Show the first image/grid, its first windows, and its reconstruction.

    Inputs use [B, H, W, C] and [B * nH * nW, M, M, C]. RGB is shown directly;
    other feature widths use channel zero. annotate is intended for small grids
    of token IDs, where spatial ordering is easy to inspect.
    """
    M = windows.shape[1]
    windows_per_image = (grid.shape[1] // M) * (grid.shape[2] // M)
    count = min(max_windows, windows_per_image)
    panels = [grid[0], *windows[:count], restored[0]]
    labels = ["Original", *[f"Window {i}" for i in range(count)], "Reconstructed"]
    figure, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3), squeeze=False)
    color_limits = dict(vmin=grid[0, ..., 0].min().item(), vmax=grid[0, ..., 0].max().item())

    for axis, panel, label in zip(axes[0], panels, labels):
        values = panel.detach().cpu().numpy()
        if values.shape[-1] not in (3, 4):
            values = values[..., 0]
        if values.ndim == 2:
            axis.imshow(values, cmap="viridis", **color_limits)
        else:
            axis.imshow(values)
        if annotate and values.ndim == 2:
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    axis.text(col, row, f"{values[row, col]:g}", ha="center", va="center",
                              color="white", bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"))
        axis.set_title(label)
        axis.axis("off")

    figure.tight_layout()
    plt.show()
    return figure
