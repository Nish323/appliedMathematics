import numpy as np
from PIL import Image
import time
import math

def calc_matrixA(figuresize, angle_division, s_division, start_angle=-90, end_angle=90):
    start_angle=math.radians(start_angle)
    end_angle=math.radians(end_angle)
    domainsizemax = figuresize / 2
    num_angles = angle_division - 1
    num_s = s_division - 1
    
    # 角度とsのサンプリング
    m = np.arange(num_angles)
    omega_aux = start_angle + ((m + 1) * (end_angle - start_angle) / angle_division)
    
    # omegaの計算 (ベクトル化)
    abs_omega_aux = np.abs(omega_aux)
    omega = np.where(abs_omega_aux <= np.pi/4, abs_omega_aux, (np.pi/2) - abs_omega_aux)
    
    cosm = np.cos(omega_aux)
    sinm = np.sin(omega_aux)
    cosp = np.cos(omega)
    sinp = np.sin(omega)
    
    n = np.arange(num_s)
    sn = (-np.sqrt(2) * domainsizemax) + ((2 * (n + 1) * np.sqrt(2) * domainsizemax) / s_division)
    
    # --- ここが重要：元のコードの i, j の並び順を再現 ---
    # 元コード: i方向(x)が外側ループ、j方向(y)が内側ループ
    # A[..., i * figuresize + j] というインデックスに対応させる
    i_idx = np.arange(figuresize)
    j_idx = np.arange(figuresize)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij') # indexing='ij' で i, j の順を維持
    
    pos_x = -domainsizemax + 0.5 + ii.flatten()
    pos_y = domainsizemax - 0.5 - jj.flatten()
    
    A = np.zeros((num_angles * num_s, figuresize**2))
    eps = 1e-10 # 0割り防止
    
    for idx_m in range(num_angles):
        # distの計算を一括で行う (num_s, figuresize^2)
        dist = np.abs(cosm[idx_m] * pos_x + sinm[idx_m] * pos_y - sn[:, np.newaxis])
        
        cond1 = (dist <= (cosp[idx_m] - sinp[idx_m]) / 2)
        cond2 = ((cosp[idx_m] - sinp[idx_m]) / 2 < dist) & (dist <= (cosp[idx_m] + sinp[idx_m]) / 2)
        
        val1 = 1 / (cosp[idx_m] + eps)
        val2 = (cosp[idx_m] + sinp[idx_m] - 2 * dist) / (2 * cosp[idx_m] * sinp[idx_m] + eps)
        
        row_start = idx_m * num_s
        row_end = (idx_m + 1) * num_s
        
        A[row_start:row_end, :] = np.where(cond1, val1, np.where(cond2, val2, 0))
        
    return A

def tikhonov(img_array, angle_division, s_division, alpha, sg_error, start_angle, end_angle):
    figuresize = img_array.shape[0]
    
    # 1. 行列Aの生成
    A = calc_matrixA(figuresize, angle_division, s_division, start_angle, end_angle)
    
    # 2. 元コードのベクトル f の作り方を再現
    # 元コード: f[j * figuresize + i] = Ffigure[i, j]
    # つまり、画像の行(y)と列(x)を入れ替えて平坦化している
    f = img_array.T.flatten().reshape(-1, 1)
    
    # 3. サイノグラム生成とノイズ付与
    sinogram = A @ f
    sinogram_norm = np.linalg.norm(sinogram, ord=2)
    
    # 再現性を高めるため、シード値を固定しても良いですが、ここではrd.uniformを再現
    noise_aux = np.random.uniform(-1, 1, sinogram.shape)
    noise = (sg_error * sinogram_norm / np.linalg.norm(noise_aux, ord=2)) * noise_aux
    sinogram_noisy = sinogram + noise
    
    # 4. 解く
    left = A.T @ A + alpha * np.eye(figuresize**2)
    right = A.T @ sinogram_noisy
    ft = np.linalg.solve(left, right)
    
    # 5. 画像の形に戻す (元コードの imgt[i, j] = ft[j * figuresize + i] を再現)
    reconstructed_img = ft.reshape(figuresize, figuresize).T
    
    return reconstructed_img

####################################################
img_pil = Image.open('img/image.png').convert('L')
img_pil = img_pil.resize((128, 128)) 
aa = np.array(img_pil)
bb = 80 # 角分割数
cc = 120 # 位置分割数
dd = 100.00 # 正則化パラメータ
rerror = 0.01 #誤差レベル
start_angle = -10
end_angle = 10
reconstruct = tikhonov(aa,bb, cc, dd, rerror, start_angle, end_angle)
pil_img = Image.fromarray(reconstruct.astype(np.uint8)) #画像として出力
pil_img.save('img/limited/image_optimized_promo.png')