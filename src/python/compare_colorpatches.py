"""
Color Patch Comparison Tool

Compare reference ColorChecker patches with detected patches from an image.
Uses colour-science and colour-checker-detection libraries.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import colour
from colour_checker_detection import detect_colour_checkers_segmentation
import cv2


def load_reference_from_lab_file(filepath: str, illuminant: str = "D65") -> tuple:
    """
    Load reference colours from a Lab values file.

    Parameters
    ----------
    filepath : str
        Path to the file containing Lab values (one patch per line, L,a,b format)
    illuminant : str
        Reference illuminant for Lab to XYZ conversion (default: "D65")

    Returns
    -------
    tuple
        (patch_names, sRGB_values) - patch names and their sRGB values
    """
    # Standard ColorChecker patch names
    default_patch_names = [
        "dark skin",
        "light skin",
        "blue sky",
        "foliage",
        "blue flower",
        "bluish green",
        "orange",
        "purplish blue",
        "moderate red",
        "purple",
        "yellow green",
        "orange yellow",
        "blue",
        "green",
        "red",
        "yellow",
        "magenta",
        "cyan",
        "white 9.5 (.05 D)",
        "neutral 8 (.23 D)",
        "neutral 6.5 (.44 D)",
        "neutral 5 (.70 D)",
        "neutral 3.5 (1.05 D)",
        "black 2 (1.5 D)",
    ]

    # Load Lab values from file
    Lab_values = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                values = [float(v.strip()) for v in line.split(",")]
                Lab_values.append(values)

    Lab_values = np.array(Lab_values)

    # Generate patch names
    n_patches = len(Lab_values)
    if n_patches == 24:
        patch_names = default_patch_names
    else:
        patch_names = [f"Patch {i + 1}" for i in range(n_patches)]

    # Convert Lab to XYZ
    XYZ_values = colour.Lab_to_XYZ(Lab_values)

    # Convert XYZ to sRGB
    sRGB_values = colour.XYZ_to_sRGB(XYZ_values)

    # Clip to valid range [0, 1]
    sRGB_values = np.clip(sRGB_values, 0, 1)

    return patch_names, sRGB_values, Lab_values


def create_simple_sidebyside(
    reference_colours: np.ndarray,
    detected_colours: np.ndarray,
    patch_size: int = 50,
    patch_gap: int = 4,
    bg_color: tuple = (255, 255, 255),
    output_path: str = None,
) -> np.ndarray:
    """
    Create a simple side-by-side comparison image with patches directly adjacent.
    Reference and detected patches are touching, with gaps between different patches.

    Parameters
    ----------
    reference_colours : np.ndarray
        Reference sRGB values (N, 3)
    detected_colours : np.ndarray
        Detected sRGB values (N, 3)
    patch_size : int
        Size of each colour patch in pixels
    patch_gap : int
        Gap between different patch pairs
    bg_color : tuple
        Background color (R, G, B) for gaps
    output_path : str, optional
        If provided, save the image to this path

    Returns
    -------
    np.ndarray
        RGB image array (uint8)
    """
    n_patches = len(reference_colours)
    n_cols = 6
    n_rows = 4

    # Calculate image dimensions
    # Each patch pair: ref + det touching (no gap)
    pair_width = patch_size * 2
    img_width = n_cols * pair_width + (n_cols - 1) * patch_gap
    img_height = n_rows * patch_size + (n_rows - 1) * patch_gap

    # Create image array with background color
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    img[:, :] = bg_color

    for i in range(n_patches):
        row = i // n_cols
        col = i % n_cols

        # Calculate pixel positions (with gaps between patches)
        x_start = col * (pair_width + patch_gap)
        y_start = row * (patch_size + patch_gap)

        # Reference patch (left)
        ref_colour = np.clip(reference_colours[i], 0, 1)
        img[y_start : y_start + patch_size, x_start : x_start + patch_size] = (
            ref_colour * 255
        ).astype(np.uint8)

        # Detected patch (right) - touching reference
        det_colour = np.clip(detected_colours[i], 0, 1)
        x_det = x_start + patch_size  # No gap
        img[y_start : y_start + patch_size, x_det : x_det + patch_size] = (
            det_colour * 255
        ).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

    return img


def create_single_checker(
    colours: np.ndarray, patch_size: int = 50, output_path: str = None
) -> np.ndarray:
    """
    Create a single ColorChecker image from colour values.

    Parameters
    ----------
    colours : np.ndarray
        sRGB values (N, 3)
    patch_size : int
        Size of each colour patch in pixels
    output_path : str, optional
        If provided, save the image to this path

    Returns
    -------
    np.ndarray
        RGB image array (uint8)
    """
    n_patches = len(colours)
    n_cols = 6
    n_rows = 4

    img_width = n_cols * patch_size
    img_height = n_rows * patch_size

    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    for i in range(n_patches):
        row = i // n_cols
        col = i % n_cols

        x_start = col * patch_size
        y_start = row * patch_size

        colour = np.clip(colours[i], 0, 1)
        img[y_start : y_start + patch_size, x_start : x_start + patch_size] = (
            colour * 255
        ).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Saved: {output_path}")

    return img


def get_reference_colours(checker_name: str = "ColorChecker 2005") -> tuple:
    """
    Get reference sRGB values for a ColorChecker.

    Parameters
    ----------
    checker_name : str
        Name of the ColorChecker (e.g., "ColorChecker 2005", "ColorChecker24 - After November 2014")

    Returns
    -------
    tuple
        (colour_names, sRGB_values) - patch names and their reference sRGB values
    """
    # Get the colour checker data
    colour_checker = colour.CCS_COLOURCHECKERS[checker_name]

    # Extract patch names and xyY values
    patch_names = list(colour_checker.data.keys())
    xyY_values = np.array(list(colour_checker.data.values()))

    # Convert xyY to XYZ
    XYZ_values = colour.xyY_to_XYZ(xyY_values)

    # Convert XYZ to sRGB (using D65 illuminant)
    sRGB_values = colour.XYZ_to_sRGB(XYZ_values)

    # Clip to valid range [0, 1]
    sRGB_values = np.clip(sRGB_values, 0, 1)

    return patch_names, sRGB_values


def detect_patches_from_image(image: np.ndarray) -> np.ndarray:
    """
    Detect ColorChecker patches from an image.

    Parameters
    ----------
    image : np.ndarray
        Input image in sRGB (values in [0, 1] range)

    Returns
    -------
    np.ndarray
        Detected sRGB values for each patch
    """
    # Detect colour checkers
    swatches = detect_colour_checkers_segmentation(image)

    if len(swatches) == 0:
        raise ValueError("No ColorChecker detected in the image")

    # Use the first detected checker
    detected_colours = swatches[0]

    # Reshape to 24 patches if needed (assuming Classic ColorChecker)
    if detected_colours.shape[0] == 24:
        return detected_colours
    else:
        # Flatten and reshape
        return detected_colours.reshape(-1, 3)[:24]


def create_comparison_image(
    reference_colours: np.ndarray,
    detected_colours: np.ndarray,
    patch_names: list = None,
    patch_size: int = 60,
    gap: int = 8,
    figsize: tuple = (16, 10),
    title: str = "Color Patch Comparison: Reference vs Detected",
) -> plt.Figure:
    """
    Create a comparison image showing reference and detected colours side by side.
    Layout matches the standard ColorChecker arrangement (6 cols x 4 rows).

    Parameters
    ----------
    reference_colours : np.ndarray
        Reference sRGB values (N, 3)
    detected_colours : np.ndarray
        Detected sRGB values (N, 3)
    patch_names : list, optional
        Names for each patch
    patch_size : int
        Size of each colour patch
    gap : int
        Gap between reference and detected patches
    figsize : tuple
        Figure size
    title : str
        Figure title

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_patches = len(reference_colours)

    # ColorChecker layout: 6 cols x 4 rows
    n_cols = 6
    n_rows = 4

    # Layout parameters
    patch_gap = 15  # Gap between patch pairs

    # Calculate canvas size
    pair_width = patch_size * 2 + gap
    canvas_width = n_cols * pair_width + (n_cols - 1) * patch_gap
    canvas_height = n_rows * patch_size + (n_rows - 1) * patch_gap

    fig, ax = plt.subplots(figsize=figsize)

    label_space = 30

    ax.set_xlim(-20, canvas_width + 20)
    ax.set_ylim(-20, canvas_height + label_space + 40)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Draw legend at top
    legend_y = canvas_height + label_space + 5
    ax.text(
        pair_width / 2,
        legend_y,
        "Reference",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        pair_width / 2 + patch_size + gap,
        legend_y,
        "Detected",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    for i in range(n_patches):
        row = i // n_cols
        col = i % n_cols

        # Calculate position (top-left origin, going right then down)
        x = col * (pair_width + patch_gap)
        y = canvas_height - (row + 1) * patch_size - row * patch_gap

        # Reference patch (left)
        ref_colour = np.clip(reference_colours[i], 0, 1)
        rect_ref = Rectangle(
            (x, y),
            patch_size,
            patch_size,
            facecolor=ref_colour,
            edgecolor="#333333",
            linewidth=1,
        )
        ax.add_patch(rect_ref)

        # Detected patch (right)
        det_colour = np.clip(detected_colours[i], 0, 1)
        rect_det = Rectangle(
            (x + patch_size + gap, y),
            patch_size,
            patch_size,
            facecolor=det_colour,
            edgecolor="#333333",
            linewidth=1,
        )
        ax.add_patch(rect_det)

        # Add patch name if provided
        if patch_names is not None and i < len(patch_names):
            name = patch_names[i]
            if len(name) > 12:
                name = name[:11] + "…"
            ax.text(
                x + pair_width / 2,
                y + patch_size + 3,
                name,
                ha="center",
                va="bottom",
                fontsize=6,
                color="#555555",
            )

    plt.tight_layout()
    return fig


def create_detailed_comparison(
    reference_colours: np.ndarray,
    detected_colours: np.ndarray,
    patch_names: list = None,
    figsize: tuple = (16, 10),
    reference_Lab: np.ndarray = None,
    detected_Lab: np.ndarray = None,
    patch_size: int = 60,
    show_labels: bool = True,
) -> plt.Figure:
    """
    Create a detailed comparison with colour difference metrics.
    Layout matches the standard ColorChecker arrangement (6 cols x 4 rows).

    Parameters
    ----------
    reference_colours : np.ndarray
        Reference sRGB values (N, 3)
    detected_colours : np.ndarray
        Detected sRGB values (N, 3)
    patch_names : list, optional
        Names for each patch
    figsize : tuple
        Figure size
    reference_Lab : np.ndarray, optional
        Reference Lab values (N, 3). If provided, use these instead of converting from sRGB.
    detected_Lab : np.ndarray, optional
        Detected Lab values (N, 3). If provided, use these instead of converting from sRGB.
    patch_size : int
        Size of each colour patch in pixels
    show_labels : bool
        Whether to show patch names

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    n_patches = len(reference_colours)

    # Calculate Delta E (CIE2000)
    if reference_Lab is not None:
        ref_Lab = reference_Lab
    else:
        ref_Lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(reference_colours))

    if detected_Lab is not None:
        det_Lab = detected_Lab
    else:
        det_Lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(detected_colours))

    delta_E = colour.delta_E(ref_Lab, det_Lab, method="CIE 2000")

    # ColorChecker layout: 6 cols x 4 rows
    n_cols = 6
    n_rows = 4

    # Layout parameters
    gap = 8  # Gap between ref and detected patches
    patch_gap = 15  # Gap between patch pairs

    # Calculate canvas size
    pair_width = patch_size * 2 + gap
    canvas_width = n_cols * pair_width + (n_cols - 1) * patch_gap
    canvas_height = n_rows * patch_size + (n_rows - 1) * patch_gap

    fig, ax = plt.subplots(figsize=figsize)

    # Add extra space for labels and stats
    label_space = 40 if show_labels else 10
    stats_space = 60
    total_height = canvas_height + label_space + stats_space + 60

    ax.set_xlim(-20, canvas_width + 20)
    ax.set_ylim(-stats_space - 20, canvas_height + label_space + 40)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.set_title(
        "Color Patch Comparison: Reference | Detected\n(ColorChecker Layout)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Draw legend at top
    legend_y = canvas_height + label_space + 10
    ax.text(
        pair_width / 2,
        legend_y,
        "Ref",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        pair_width / 2 + patch_size + gap,
        legend_y,
        "Det",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    for i in range(n_patches):
        row = i // n_cols
        col = i % n_cols

        # Calculate position (top-left origin, going right then down)
        x = col * (pair_width + patch_gap)
        y = canvas_height - (row + 1) * patch_size - row * patch_gap

        # Reference patch (left)
        ref_colour = np.clip(reference_colours[i], 0, 1)
        rect_ref = Rectangle(
            (x, y),
            patch_size,
            patch_size,
            facecolor=ref_colour,
            edgecolor="#333333",
            linewidth=1,
        )
        ax.add_patch(rect_ref)

        # Detected patch (right)
        det_colour = np.clip(detected_colours[i], 0, 1)
        rect_det = Rectangle(
            (x + patch_size + gap, y),
            patch_size,
            patch_size,
            facecolor=det_colour,
            edgecolor="#333333",
            linewidth=1,
        )
        ax.add_patch(rect_det)

        # Delta E label (between patches, at bottom)
        de_color = (
            "green" if delta_E[i] < 2 else ("orange" if delta_E[i] < 5 else "red")
        )
        ax.text(
            x + patch_size + gap / 2,
            y - 3,
            f"ΔE={delta_E[i]:.1f}",
            ha="center",
            va="top",
            fontsize=7,
            color=de_color,
            fontweight="bold",
        )

        # Patch name label
        if show_labels and patch_names is not None and i < len(patch_names):
            # Truncate long names
            name = patch_names[i]
            if len(name) > 12:
                name = name[:11] + "…"
            ax.text(
                x + pair_width / 2,
                y + patch_size + 3,
                name,
                ha="center",
                va="bottom",
                fontsize=6,
                color="#555555",
            )

    # Summary statistics box at bottom
    stats_text = (
        f"Mean ΔE*00: {np.mean(delta_E):.2f}   |   "
        f"Max: {np.max(delta_E):.2f}   |   "
        f"Min: {np.min(delta_E):.2f}   |   "
        f"Std: {np.std(delta_E):.2f}"
    )
    ax.text(
        canvas_width / 2,
        -30,
        stats_text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )

    plt.tight_layout()
    return fig


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file.

    Parameters
    ----------
    image_path : str
        Path to the image file

    Returns
    -------
    np.ndarray
        Image in sRGB (values in [0, 1] range)
    """
    # Load with OpenCV (BGR format)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    # Convert BGR to RGB and normalize to [0, 1]
    if img.dtype == np.uint8:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 65535.0
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# =============================================================================
# Example usage / Demo
# =============================================================================
if __name__ == "__main__":
    patch_names, ref_sRGB, ref_Lab = load_reference_from_lab_file("data/ref_lab.txt")

    # Load your image and detect patches
    image = load_image("data/corrected_image.png")
    detected_colours = detect_patches_from_image(image)

    # Create comparison (using original Lab for accurate Delta E)
    fig = create_detailed_comparison(
        ref_sRGB, detected_colours, patch_names, reference_Lab=ref_Lab
    )
    fig.savefig("data/my_comparison.png")

    fig2 = create_comparison_image(
        ref_sRGB, detected_colours, patch_names, title="Simple Comparison"
    )
    fig2.savefig("data/my_simple_comparison.png")

    fig3 = create_simple_sidebyside(
        ref_sRGB, detected_colours, output_path="data/my_sidebyside.png"
    )
