import numpy as np
import colour
import matplotlib.pyplot as plt
import cv2
from colour_checker_detection import (
    detect_colour_checkers_segmentation,
    detect_colour_checkers_inference,
)


def extract_swatches(linear_image, debug: bool = False) -> np.ndarray:
    """
    カラーチャートを検出して正確なLinear RGB値を抽出する関数

    Parameters:
        linear_image: rawpyで現像した (H, W, 3) の 0.0-1.0 float Linear画像
    Returns:
        swatches_linear: (24, 3) の正規化されたLinear RGB配列
    """

    # ---------------------------------------------------------
    # 1. 検出用プロキシ画像の作成 (Exposure Gain法)
    # ---------------------------------------------------------
    # 暗いリニアデータのままだと検出できないため、線形性を保ったまま明るくします。
    # 平均輝度を見てゲインを自動調整、あるいは固定値(例: 5.0)を使用
    mean_val = np.mean(linear_image)
    if mean_val < 0.05:  # とても暗い場合
        mean_val = max(mean_val, 1e-6)  # ゼロ除算防止
        gain = 0.18 / mean_val  # 中性グレー(0.18)付近まで持ち上げる
        print(f"画像が暗いため、Gain {gain:.2f}倍 で検出します")
    else:
        gain = 1.0

    # 検出用画像 (クリップしないとエラーになる場合があるためclip)
    proxy_image = np.clip(linear_image * gain, 0, 1.0)

    # ---------------------------------------------------------
    # 2. カラーチャート検出 (Segmentation)
    # ---------------------------------------------------------
    # additional_data=True にすると、補正後の画像やマスク情報が取れます
    detection_results = detect_colour_checkers_segmentation(
        proxy_image,
        additional_data=True,
        show=debug,
    )

    if len(detection_results) == 0:
        print("カラーチャートの検出に失敗しました")
        return None

    # 最初の検出結果を取得
    colour_checker_data = detection_results[0]

    # valuesプロパティからデータをアンパック (サンプルコードと同じ処理)
    # swatch_colours: 抽出されたRGB値 (24, 3)
    # swatch_masks: 各パッチの境界ボックス [y_min, y_max, x_min, x_max]
    # colour_checker_image: 傾き補正後のチャート画像
    colour_checker_data = detection_results[0]
    # 返り値が3つ以上あっても対応できるように修正
    values = colour_checker_data.values
    swatch_colours = values[0]  # 抽出された色
    swatch_masks = values[1]  # マスク座標
    colour_checker_image = values[2]  # 補正後の画像

    # ---------------------------------------------------------
    # 3. データの復元 (Linearに戻す)
    # ---------------------------------------------------------
    # 抽出された swatch_colours は proxy_image (gain倍) の値なので、
    # gain で割って元のスケールに戻します。
    swatches_linear = swatch_colours / gain

    print("抽出成功: Linear RGB値を取得しました")

    # ---------------------------------------------------------
    # 4. 可視化 (サンプルコードのロジックを活用)
    # ---------------------------------------------------------
    # 確認用: 検出されたパッチがどこかを表示
    masks_i = np.zeros(colour_checker_image.shape)
    for i, mask in enumerate(swatch_masks):
        # mask = [y_min, y_max, x_min, x_max]
        masks_i[mask[0] : mask[1], mask[2] : mask[3], ...] = 1

    # 表示は見やすいようにガンマ補正(cctf_encoding)をかける
    vis_image = colour.cctf_encoding(
        np.clip(colour_checker_image + masks_i * 0.25, 0, 1)
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(vis_image)
    plt.title(f"Detected & Rectified Chart (Gain: {gain:.2f})")
    plt.axis("off")
    plt.show()

    return swatches_linear


# --- 実行例 ---
# if __name__ == "__main__":
#     from raw_to_png import develop_raw
#     # RAW画像をリニア現像して取得
#     raw_path = "test_images/colorchecker.CR3"  # 例:
#     img = develop_raw(raw_path)
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 65535.0  # 0.0-1.0に正規化
#     img_float = img_rgb.astype(np.float32)
#     result = extract_swatches(img_float)

#     if result is not None:
#         print("\n--- 先頭のパッチ (Dark Skin) のLinear値 ---")
#         print(result[0])
