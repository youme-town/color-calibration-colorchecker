"""
Color Correction Module using Root-Polynomial Color Correction (RPCC)
(Root-Polynomial Color Correction を用いた色補正モジュール)

This module provides functions for color correction of camera images
using the RPCC method with ColorChecker reference data.
(カラーチェッカーの基準データを用いたRPCC法によるカメラ画像の色補正機能を提供します)
"""

import colour
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal
from raw_to_png import develop_raw
from extract_colorchecker import extract_swatches
import cv2


def crop_img_interactively(img: np.ndarray, window_size: Tuple[int, int]) -> np.ndarray:
    """
    Display an image and crop a region selected by mouse drag.
    (画像を表示して、マウスでドラッグした範囲をクロップする関数)

    Parameters
    ----------
    img : np.ndarray
        Input image as a numpy array with shape (H, W, 3) in RGB format.
        (RGB形式の入力画像、形状は (H, W, 3))
    window_size : Tuple[int, int]
        Display window size as (width, height).
        (表示ウィンドウのサイズ (幅, 高さ))

    Returns
    -------
    np.ndarray
        Cropped image based on the user-selected region of interest.
        (ユーザーが選択した領域でクロップされた画像)
    """
    cv2.namedWindow(
        "Crop Image - Select ROI and press ENTER or SPACE", cv2.WINDOW_NORMAL
    )
    cv2.resizeWindow(
        "Crop Image - Select ROI and press ENTER or SPACE",
        window_size[0],
        window_size[1],
    )
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    roi = cv2.selectROI(
        "Crop Image - Select ROI and press ENTER or SPACE", bgr_img, False
    )
    cv2.destroyAllWindows()
    x, y, w, h = roi
    cropped_img = img[y : y + h, x : x + w]
    return cropped_img


def load_lab_reference(filepath: str) -> np.ndarray:
    """
    Load reference CIE Lab data for a color chart from a file.
    (カラーチャートの基準CIE Labデータをファイルから読み込む関数)

    Parameters
    ----------
    filepath : str
        Path to the text file containing Lab data in CSV format.
        (Lab値が保存されたCSV形式のテキストファイルのパス)

    Returns
    -------
    np.ndarray
        Reference Lab values as a (24, 3) array of float32.
        (基準Lab値、形状 (24, 3) のfloat32配列)
    """
    reference_lab = np.loadtxt(
        filepath,
        delimiter=",",
        dtype=np.float32,
    )
    return reference_lab


def apply_rpcc_correction(
    linear_image: np.ndarray,
    RPCC_MATRIX: np.ndarray,
    degree: Literal[1, 2, 3] = 2,
) -> np.ndarray:
    """
    Apply Root-Polynomial Color Correction (RPCC) to an image.
    (Root-Polynomial Color Correction (RPCC) を画像に適用する関数)

    This function expands the RGB values using root-polynomial expansion
    and applies the pre-computed RPCC matrix to convert camera RGB to XYZ.
    (RGB値をルート多項式展開で拡張し、事前計算されたRPCC行列を適用して
    カメラRGBをXYZに変換します)

    Parameters
    ----------
    linear_image : np.ndarray
        Linear camera image with shape (H, W, 3), values in range [0.0, 1.0].
        (リニアカメラ画像、形状 (H, W, 3)、値の範囲は [0.0, 1.0])
    RPCC_MATRIX : np.ndarray
        Pre-computed RPCC transformation matrix with shape (3, N),
        where N depends on the degree (6 for degree=2, 13 for degree=3).
        (事前計算されたRPCC変換行列、形状 (3, N)、Nは次数に依存)
    degree : Literal[1, 2, 3], optional
        Polynomial degree for expansion. Default is 2.
        - 1: Linear (R, G, B)
        - 2: Quadratic with cross terms (R, G, B, sqrt(RG), sqrt(GB), sqrt(BR))
        - 3: Cubic with additional terms
        (展開の多項式次数。デフォルトは2)

    Returns
    -------
    np.ndarray
        Color-corrected image in XYZ color space with shape (H, W, 3),
        values clipped to range [0.0, 1.0].
        (XYZ色空間の補正済み画像、形状 (H, W, 3)、値は [0.0, 1.0] にクリップ)

    Notes
    -----
    The RPCC matrix expects expanded RGB input, so this function internally
    performs the same polynomial expansion on the image data.
    (RPCC行列は拡張されたRGB入力を期待するため、この関数は内部で
    画像データに対して同じ多項式展開を実行します)
    """
    # Expand image data: [R, G, B] -> [R, G, B, sqrt(RG), sqrt(GB), sqrt(BR)]
    # (画像データの拡張処理)
    expanded_image = colour.characterisation.polynomial_expansion_Finlayson2015(
        linear_image, degree=degree, root_polynomial_expansion=True
    )

    print(f"Expanded Image Shape: {expanded_image.shape}")

    # Matrix multiplication: XYZ = M @ Expanded_RGB
    # (行列演算)
    corrected_xyz = colour.algebra.vector_dot(RPCC_MATRIX, expanded_image)

    # Clip values to valid range (noise reduction)
    # (有効範囲にクリップ、ノイズ対策)
    corrected_xyz = np.clip(corrected_xyz, 0.0, 1.0)

    return corrected_xyz


# --- Main execution example (実行例) ---
def main():
    # =================================================
    # RPCC Matrix Calculation Example
    # (RPCC行列の計算例)
    # =================================================
    reference_lab = load_lab_reference("data/ref_lab.txt")  # ColorChecker Lab reference
    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    reference_xyz = colour.Lab_to_XYZ(reference_lab, illuminant=D65)
    linear_image = develop_raw("data/IMG_9036.CR3")
    linear_image = crop_img_interactively(linear_image, (1920, 1080))
    linear_image = (
        linear_image.astype(np.float32) / 65535.0
    )  # 16bit -> [0.0, 1.0] float
    linear_image = np.clip(linear_image, 0.0, 1.0)

    measured_swatches = extract_swatches(linear_image)

    if measured_swatches is None:
        raise ValueError("ColorChecker swatches could not be extracted.")

    M_RPCC = colour.characterisation.matrix_colour_correction_Finlayson2015(
        M_T=measured_swatches,
        M_R=reference_xyz,
        degree=2,
        root_polynomial_expansion=True,  # True for RPCC, False for standard polynomial
    )
    save_path = "data/M_RPCC.npy"
    np.save(save_path, M_RPCC)
    print(f"Saved RPCC matrix to {save_path}")

    print("MRPCC Matrix:")
    print(M_RPCC)

    # =================================================
    # Image Application Example
    # (画像への適用例)
    # =================================================

    # Load image that needs color correction
    # (色補正が必要な画像を読み込み)
    uncorrected_image = develop_raw("data/IMG_9036.CR3")
    uncorrected_image = uncorrected_image.astype(np.float32) / 65535.0
    uncorrected_image = np.clip(uncorrected_image, 0.0, 1.0)
    print(
        "[Max/Min] Uncorrected Image:",
        np.max(uncorrected_image),
        np.min(uncorrected_image),
    )

    # Apply RPCC correction
    # (RPCC補正を実行)
    result_xyz_img = apply_rpcc_correction(uncorrected_image, M_RPCC, degree=2)

    # Convert XYZ to sRGB for display
    # (表示用にXYZからsRGBに変換)
    result_img = colour.XYZ_to_sRGB(result_xyz_img)
    result_img = np.clip(result_img, 0.0, 1.0)
    viz_result = (result_img * 255).astype(np.uint8)

    plt.imshow(viz_result)
    plt.title("RPCC Corrected Image")
    plt.axis("off")
    plt.show()
    cv2.imwrite("data/corrected_image.png", cv2.cvtColor(viz_result, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
