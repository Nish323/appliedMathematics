import sys
sys.path.append('src')

from tikhonov_optimized import run_reconstruction
import time

# パラメータセットのリスト
# 各タプルは (bb, cc, dd, rerror) の形式
parameters = [
    # --- 正則化パラメータの極端な振り（対数スケール） ---
    (50, 120, 10.0, 0.01),   # 弱正則化
    (50, 120, 500.0, 0.01),  # 強正則化
    
    # --- 誤差レベルの境界条件 ---
    (50, 120, 100.0, 0.0),   # ノイズなし
    (50, 120, 100.0, 0.05),  # 強ノイズ
    
    # --- 解像度のバリエーション ---
    (30, 80, 100.0, 0.01),   # 粗い分割
    (80, 200, 100.0, 0.01),  # 細かい分割
    
    # --- 複合的な変化（ノイズに合わせて正則化を調整） ---
    (50, 120, 200.0, 0.02),  # ノイズ大に合わせて正則化を強める
    (50, 120, 50.0, 0.005),  # ノイズ小に合わせて正則化を弱める
]

def main():
    total_start = time.time()
    
    print(f"Total {len(parameter_sets)} parameter sets to process")
    print("=" * 60)
    
    for i, (bb, cc, dd, rerror) in enumerate(parameter_sets, 1):
        print(f"\n[{i}/{len(parameter_sets)}] Processing: bb={bb}, cc={cc}, dd={dd}, rerror={rerror}")
        try:
            run_reconstruction(bb, cc, dd, rerror)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"All done! Total time: {total_time:.2f}s")

if __name__ == '__main__':
    main()
