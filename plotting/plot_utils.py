import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")


def draw_multiple_loglog_slopes(
    fig,
    ax,
    origin,
    triangle_width,
    slopes,
    dashed_extra_slopes=False,
    inverted=False,
    color=None,
    label=True,
    labelcolor=None,
    fontsize_factor=0.8,
):
    """This function draws slopes or "convergence triangles" into loglog plots.

    References:
        The creation of the main triangle is from:
        https://gist.github.com/w1th0utnam3/a0189dc8a2c067ccb56e1de8c317b190

        All other functionality (insertion of extra slope lines, adaptation of label
        locations etc.) are inspired from the original reference.

    Parameters:
        fig: The figure.
        ax: The axes object to draw to.
        origin: The origin coordinates of the triangle.
        triangle_width: The width in inches of the triangle.
        slopes: The list of slopes to be drawn. That is, orders of convergence for the
            "convergence triangle(s)".
        dashed_extra_slopes: Bool value of whether the non-max slopes should be dashed
        inverted: Whether to mirror the triangle (if the 90 degree angle of the
            triangle is in the lower right or upper left).
        color: Color of the triangle edges.
        label: Whether to enable labeling of the slopes. Defaults to True.
        labelcolor: The color of the slope labels. Defaults to edge color.
        fontsize_factor: Determines the size of the slope labels relative to the
            default font size. Defaults to 0.8.

    Example:
        # Create a figure with plotted array values
            fig, ax = plt.subplots()
            ax.loglog(x, y)

        # Then call this function using fig, ax and other parameters:
            draw_multiple_loglog_slopes(
                fig,
                ax,
                origin=(x[-1], y_disp[-1]),
                triangle_width=1.0,
                slopes=[1, 2],
                labelcolor=(0.33, 0.33, 0.33),
            )
            plt.show()

    """

    zorder, alpha = 10, 0.25
    slopes = np.asarray(slopes)

    color = color or (0.25, 0.25, 0.25)
    labelcolor = labelcolor or color

    poly_kwargs = dict(
        color=color,
        linewidth=0.8 * plt.rcParams["lines.linewidth"],
    )
    label_kwargs = dict(
        color=labelcolor,
        fontsize=fontsize_factor * plt.rcParams["font.size"],
    )

    triangle_width *= -1 if inverted else 1

    # Coordinate transforms
    # Convert the origin into figure coordinates in inches
    origin_disp = ax.transData.transform(origin)
    origin_dpi = fig.dpi_scale_trans.inverted().transform(origin_disp)

    # Obtain the bottom-right corner in data coordinates
    corner_dpi = origin_dpi + triangle_width * np.array([1.0, 0.0])
    corner_disp = fig.dpi_scale_trans.transform(corner_dpi)
    corner = ax.transData.inverted().transform(corner_disp)

    x1, y1 = origin
    x2 = corner[0]
    width = x2 - x1

    # Slope geometry
    use_positive = np.any(slopes > 0)
    main_slope = max(slopes) if use_positive else min(slopes)
    log_offset = y1 / (x1**main_slope)
    y2 = log_offset * ((x1 + width) ** main_slope)

    a, b, c = origin, corner, [x2, y2]

    # Draw triangle
    ax.add_patch(
        plt.Polygon([a, b, c], fill=True, alpha=alpha, zorder=zorder, **poly_kwargs)
    )

    # Display-space centers
    a_d, b_d, c_d = map(ax.transData.transform, (a, b, c))

    bottom_center = ax.transData.inverted().transform(a_d + 0.5 * (b_d - a_d))
    right_center = ax.transData.inverted().transform(b_d + 0.5 * (c_d - b_d))

    # --- Label alignment logic ---
    sign = 1 if use_positive else -1
    va_xlabel = "top" if sign > 0 else "bottom"
    ha_ylabel = "left" if sign > 0 else "right"

    if inverted:
        va_xlabel = "bottom" if va_xlabel == "top" else "top"
        ha_ylabel = "right" if ha_ylabel == "left" else "left"

    fs = label_kwargs["fontsize"]
    offset_xlabel = [0, -0.33 * fs * sign * (-1 if inverted else 1)]
    offset_ylabel = [0.33 * fs * sign * (-1 if inverted else 1), 0]

    # Base slope label
    ax.annotate(
        "$1$",
        bottom_center,
        xytext=offset_xlabel,
        textcoords="offset points",
        ha="center",
        va=va_xlabel,
        zorder=zorder,
        **label_kwargs,
    )

    # Main slope annotation
    if label:
        dx = 0.02 * x2 * (-1 if inverted else 1)
        dy = -0.03 * y2 if not inverted else 0
        label_point = [x2 + dx, y2 + dy]

        ha = "right" if inverted else "left"
        va = "center"

        if len(slopes) == 1:
            label_point[1] = (y2 + b[1]) / 2
            va = "top"

        ax.annotate(
            f"${abs(main_slope)}$",
            label_point,
            ha=ha,
            va=va,
            zorder=zorder,
            **label_kwargs,
        )

    # Extra slopes
    if len(slopes) > 1:
        for slope in [s for s in slopes if s != main_slope]:
            y_end = (y1 / x1**slope) * ((x1 + width) ** slope)

            ax.plot(
                [x1, x2],
                [y1, y_end],
                linestyle="--" if dashed_extra_slopes else "-",
                linewidth=0.99,
                color=color,
                alpha=alpha,
                zorder=zorder - 2,
            )

            label_x = label_point[0]
            ax.annotate(
                f"${abs(slope)}$",
                [label_x, y_end],
                ha="right" if inverted else "left",
                va="center",
                zorder=zorder - 2,
                **label_kwargs,
            )
