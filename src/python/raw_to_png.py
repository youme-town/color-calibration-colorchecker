import rawpy
import numpy as np


def develop_raw(raw_path: str) -> np.ndarray:
    """
    RAW画像を読み込み、リニア現像して返す
    """
    with rawpy.imread(raw_path) as raw:
        rgb = raw.postprocess(
            # 1. ガンマ補正を無効化 (Linear)
            gamma=(1, 1),
            # 2. 自動明るさ調整を無効化 (計測値を変えないため)
            no_auto_bright=True,
            # 3. ビット深度を16bitにする
            output_bps=16,
            # 4. カメラの色空間のまま出力する
            #    rawpy.ColorSpace.sRGB (=1) にするとsRGB変換行列がかかってしまうため、
            #    rawpy.ColorSpace.raw (=0) を指定してセンサーの生の混ざり具合を保持する
            output_color=rawpy.ColorSpace.raw,
            # 5. ホワイトバランスはカメラの設定を使用する
            use_camera_wb=True,
            use_auto_wb=False,
            no_auto_scale=False,  # 自動スケールは有効のままにする
            user_sat=None,  # ハイライトクリップを行わない
        )

    return rgb


# --- 使い方 ---
# raw_path = "IMG_1234.ARW" # Sony, Canon(CR2/CR3), Nikon(NEF) 等
# linear_img = read_raw_linear(raw_path)

# 確認用表示 (OpenCVはBGRなので変換、かつ暗いので簡易的に明るくして表示)
# display_img = cv2.cvtColor((linear_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
# cv2.imshow("Linear RAW", display_img) # ガンマがかかっていないので暗く見えます
# cv2.waitKey(0)
# cv2.destroyAllWindows()
