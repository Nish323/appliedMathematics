import sys
sys.path.append('src/limited')

from tikhonov_optimized_limited_promo import run_reconstruction
import time

# パラメータセットのリスト (bb, cc, dd, rerror, start_angle, end_angle)
parameter_sets = [
    # --- グループ1: 角度スパンの影響（正則化とノイズは固定） ---
    # スパンが広がるにつれ、物体の「横方向」の輪郭が復活する様子を観察
    (80, 120, 100.0, 0.01, -10, 10),   # 20度：極端な制限
    (80, 120, 100.0, 0.01, -30, 30),   # 60度：一般的制限
    (80, 120, 100.0, 0.01, -45, 45),   # 90度：直角範囲
    (80, 120, 100.0, 0.01, -90, 90),   # 180度：理想的なフルスキャン

    # 角度が足りない時、alphaを大きくするとノイズは減るが、画像がどれだけ「ぼける」か
    (80, 120, 1.0,   0.01, -20, 20),   # 超弱：ノイズに非常に敏感
    (80, 120, 100.0, 0.01, -20, 20),   # 中間
    (80, 120, 1000.0,0.01, -20, 20),   # 超強：形状が滑らか（鈍く）なる

    # 合計角度が同じ20度でも、範囲が異なると「消えるエッジ」の方向が変わることを確認
    (80, 120, 100.0, 0.01, -10, 10),   # 中心
    (80, 120, 100.0, 0.01, 0, 20),     # 片側
    (80, 120, 100.0, 0.01, 70, 90),    # 垂直に近い（横方向のエッジが強調される）
]

def main():
    total_start = time.time()
    
    print(f"Total {len(parameter_sets)} parameter sets to process")
    print("=" * 60)
    
    for i, (bb, cc, dd, rerror, start_angle, end_angle) in enumerate(parameter_sets, 1):
        print(f"\n[{i}/{len(parameter_sets)}] Processing: bb={bb}, cc={cc}, dd={dd}, rerror={rerror}, angle={start_angle}~{end_angle}")
        try:
            run_reconstruction(bb, cc, dd, rerror, start_angle, end_angle)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"All done! Total time: {total_time:.2f}s")

if __name__ == '__main__':
    main()
