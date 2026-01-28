import sys
sys.path.append('src')

from tikhonov_optimized import run_reconstruction
import time

# パラメータセットのリスト
# 各タプルは (bb, cc, dd, rerror) の形式
parameter_sets = [
    # 角分割数、位置分割数、正則化パラメータ、誤差レベル
    (50, 120, 100.0, 0.01),
    (50, 120, 50.0, 0.01),
    (50, 120, 200.0, 0.01),
    (50, 120, 100.0, 0.005),
    (50, 120, 100.0, 0.02),
    (40, 100, 100.0, 0.01),
    (60, 140, 100.0, 0.01),
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
