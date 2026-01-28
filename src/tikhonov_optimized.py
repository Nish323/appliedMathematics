import numpy as np
from PIL import Image
import time

def calc_matrixA_optimized(figuresize, angle_division, s_division):
    domainsizemax = figuresize / 2
    num_angles = angle_division - 1
    num_s = s_division - 1
    
    # 角度とsのサンプリング
    m = np.arange(1, angle_division)
    omega_aux = (-np.pi / 2) + (m * np.pi / angle_division)
    
    # omegaの計算 (ベクトル化)
    abs_omega_aux = np.abs(omega_aux)
    omega = np.where(abs_omega_aux <= np.pi/4, abs_omega_aux, (np.pi/2) - abs_omega_aux)
    
    cosm = np.cos(omega_aux)
    sinm = np.sin(omega_aux)
    cosp = np.cos(omega)
    sinp = np.sin(omega)
    
    n = np.arange(1, s_division)
    sn = (-np.sqrt(2) * domainsizemax) + ((2 * n * np.sqrt(2) * domainsizemax) / s_division)
    
    # 画像座標のグリッド作成 (i: x方向, j: y方向)
    i_idx = np.arange(figuresize)
    j_idx = np.arange(figuresize)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')
    
    # 座標の物理値
    pos_x = -domainsizemax + 0.5 + ii.flatten()
    pos_y = domainsizemax - 0.5 - jj.flatten()
    
    A = np.zeros((num_angles * num_s, figuresize**2))
    
    for idx_m in range(num_angles):
        # distの計算を一括で行う (1, figuresize^2)
        dist = np.abs(cosm[idx_m] * pos_x + sinm[idx_m] * pos_y - sn[:, np.newaxis])
        
        # 条件分岐をマスク処理で高速化
        cond1 = (dist <= (cosp[idx_m] - sinp[idx_m]) / 2)
        cond2 = ((cosp[idx_m] - sinp[idx_m]) / 2 < dist) & (dist <= (cosp[idx_m] + sinp[idx_m]) / 2)
        
        eps = 1e-10
        denom = 2 * cosp[idx_m] * sinp[idx_m]
        val2 = (cosp[idx_m] + sinp[idx_m] - 2 * dist) / (denom + eps)
        
        row_start = idx_m * num_s
        row_end = (idx_m + 1) * num_s
        
        # Aに行列を流し込む
        A[row_start:row_end, :] = np.where(cond1, 1/cosp[idx_m], np.where(cond2, val2, 0))
        
    return A

def tikhonov_optimized(img_array, angle_division, s_division, alpha, sg_error):
    figuresize = img_array.shape[0]
    
    # 行列Aの生成
    A = calc_matrixA_optimized(figuresize, angle_division, s_division)
    
    # 画像のベクトル化
    # 元のコードの f[j*figuresize + i] = Ffigure[i, j] は Ffigure.T.flatten() と等価
    f = img_array.T.flatten().reshape(-1, 1)
    
    # 観測データ)の生成
    sinogram = A @ f
    sinogram_norm = np.linalg.norm(sinogram, ord=2)
    
    # 誤差の付与
    noise = np.random.uniform(-1, 1, sinogram.shape)
    noise = (sg_error * sinogram_norm / np.linalg.norm(noise, ord=2)) * noise
    sinogram_noisy = sinogram + noise
    
    # チコノフ正則化 (正規方程式の解)
    print("Solving Reconstruction...")
    # (A^T A + alpha * I) f = A^T b
    left = A.T @ A + alpha * np.eye(figuresize**2)
    right = A.T @ sinogram_noisy
    ft = np.linalg.solve(left, right)
    
    # 画像の形に戻す
    reconstructed_img = ft.reshape(figuresize, figuresize).T
    return reconstructed_img

# --- 実行部分 ---
start_time = time.time()

img_pil = Image.open('img/image.png').convert('L')
img_pil = img_pil.resize((128, 128)) 
aa = np.array(img_pil)
bb = 80 # 角分割数
cc = 120 # 位置分割数
dd = 100.00 # 正則化パラメータ
rerror = 0.01 #誤差レベル

reconstruct = tikhonov_optimized(aa, bb, cc, dd, rerror)

# クリッピング (0-255の範囲に収める)
pil_img = Image.fromarray(reconstruct.astype(np.uint8)) #画像として出力
pil_img.save('img/image_er001.png')