import colour
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Literal
from raw_to_png import develop_raw
from extract_colorchecker import extract_swatches
import cv2


def load_lab_reference(filepath: str) -> np.ndarray:
    """
    カラーチャートの基準XYZデータをファイルから読み込む関数

    Parameters:
        filepath: XYZデータが保存されたテキストファイルのパス
    Returns:
        reference_xyz: (24, 3) の正規化されたXYZ配列
    """
    reference_xyz = np.loadtxt(
        filepath,
        delimiter=",",
        dtype=np.float32,
    )
    return reference_xyz


def apply_rpcc_correction(
    linear_image: np.ndarray,
    RPCC_MATRIX: np.ndarray,
    degree: Literal[1, 2, 3] = 2,
) -> np.ndarray:
    """
    Root-Polynomial Color Correction (RPCC) を適用する関数

    Parameters:
        linear_image: カメラ画像 (H, W, 3) 0.0-1.0
        measured_swatches: 画像から抽出したパッチRGB (24, 3)
        reference_xyz: 基準となる正解XYZ (24, 3)
        degree: 次数 1は線形、2は2次、3は3次
    Returns:
        corrected_xyz: 補正後のXYZ画像 (H, W, 3) 0.0-1.0
    """

    # =================================================
    # 画像データへの適用
    # =================================================
    # RPCCの行列は 3x6 (または3x13) なので、
    # 3チャンネルの画像 (R,G,B) にそのまま行列を掛けることはできません。
    # 画像側のRGBも、同じルールで「拡張(Expand)」する必要があります。

    # 画像の拡張処理: [R, G, B] -> [R, G, B, sqrt(RG), sqrt(GB), sqrt(BR)]
    expanded_image = colour.characterisation.polynomial_expansion_Finlayson2015(
        linear_image, degree=degree, root_polynomial_expansion=True
    )

    print(f"Expanded Image Shape: {expanded_image.shape}")
    # (H, W, 6) などの形状になります

    # 行列演算 ( XYZ = M . Expanded_RGB )
    # expanded_image は最後の次元が項数(6 or 13)になっているので計算可能
    corrected_xyz = colour.algebra.vector_dot(RPCC_MATRIX, expanded_image)

    # クリップ処理 (ノイズ対策)
    corrected_xyz = np.clip(corrected_xyz, 0.0, 1.0)

    return corrected_xyz


# --- 実行例 ---
if __name__ == "__main__":
    reference_lab = load_lab_reference("ref_lab.txt")
    D65 = colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]
    reference_xyz = colour.Lab_to_XYZ(reference_lab, illuminant=D65)
    print("Reference XYZ:")
    print(reference_xyz)
    linear_image = develop_raw("color_checker.CR3")
    linear_image = linear_image.astype(np.float32) / 65535.0  # 16bit -> 0.0-1.0 float
    linear_image = np.clip(linear_image, 0.0, 1.0)  # クランプ
    print("[Max/Min] Linear Image:", np.max(linear_image), np.min(linear_image))
    measured_swatches = extract_swatches(linear_image)

    if measured_swatches is None:
        raise ValueError("ColorChecker swatches could not be extracted.")

    M_RPCC = colour.characterisation.matrix_colour_correction_Finlayson2015(
        M_T=measured_swatches,
        M_R=reference_xyz,
        degree=2,
        root_polynomial_expansion=True,  # これがTrueだとRPCC、Falseだと通常の多項式
    )

    print("MRPCC Matrix:")
    print(M_RPCC)

    # load image which need color correction
    uncorrected_image = develop_raw("color_checker.CR3")
    uncorrected_image = uncorrected_image.astype(np.float32) / 65535.0
    uncorrected_image = np.clip(uncorrected_image, 0.0, 1.0)  # クランプ
    print(
        "[Max/Min] Uncorrected Image:",
        np.max(uncorrected_image),
        np.min(uncorrected_image),
    )

    # RPCC を実行
    result_xyz_img = apply_rpcc_correction(uncorrected_image, M_RPCC, degree=2)

    # XYZ -> RGB変換 (sRGB)
    result_img = colour.XYZ_to_sRGB(result_xyz_img)
    result_img = np.clip(result_img, 0.0, 1.0)
    viz_result = (result_img * 255).astype(np.uint8)

    plt.imshow(viz_result)
    plt.title("RPCC Corrected Image")
    plt.axis("off")
    plt.show()
    cv2.imwrite("corrected_image.png", cv2.cvtColor(viz_result, cv2.COLOR_RGB2BGR))
