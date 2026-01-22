# coding: utf-8
# 128 * 128 の画像で計算時間12分程度

import numpy as np
from PIL import Image
import math
import random as rd
import csv

def calc_matrixA(img,angle_division, s_division): #角度は-pi/2 - pi/2, sは-sqrt(2) - sqrt(2)でとる
#angle_division, s_division はともに自然数, 刻みの個数
  figuresize = img.shape[0] #pixel 画像サイズは正方形かつ必ず縦横偶数pixとする
  domainsizemax = figuresize / 2 #figuresizeが偶数だから割り切れる
  A = np.zeros( ( ( angle_division - 1 ) * ( s_division - 1 ), figuresize ** 2 ) ) #0行列としてAを初期化
  #
  #
  for m in range(0, angle_division - 1 ): #角度の刻み
    omega_aux = ( - np.pi / 2 ) + (( m + 1 ) * np.pi / angle_division )
    if 0<= abs(omega_aux) <= np.pi/4:
      omega = abs(omega_aux)
    elif abs(omega_aux) > np.pi/4:
      omega = (np.pi/2) - abs(omega_aux)
    #print((omega,omega_aux))
    cosm = np.cos(omega_aux)
    sinm = np.sin(omega_aux)
    cosp = np.cos(omega)
    sinp = np.sin(omega)
    #print((cosm,sinm))
    for n in range(0, s_division -1): #直線の刻み
      sn = ( - np.sqrt(2) * domainsizemax ) + (( 2 * ( n + 1 ) * np.sqrt(2) * domainsizemax ) / s_division )
      #print(sn)
      for i in range(0, figuresize ): #iはx方向右へ
        for j in range(0, figuresize ): #jはy方向下へ
          dist = abs( cosm * ( - domainsizemax + (1/2) + i ) + sinm * ( domainsizemax - (1/2) - j ) - sn )
          #print(dist)
          #print((-domainsizemax + i + 1/2 , domainsizemax - j -1/2 ))
          if 0 <= dist <= ( cosp - sinp ) / 2:
            A[ m * ( s_division -1) + n , i * figuresize + j ] = 1 / cosp
          elif ( cosp - sinp ) /2 < dist <= ( cosp + sinp )/2:
            A[ m * ( s_division -1) + n , i * figuresize + j ] = ( cosp + sinp - 2 * dist ) / ( 2 * cosp * sinp )
          elif dist > ( cosp + sinp )/2:
            A[ m * ( s_division -1) + n , i * figuresize + j ] = 0
            #print(A[m+n,(figuresize -1)*j+i])
  #
  #
  return A
#
####################################################
def calc_vectorf(img, angle_division, s_division, sg_error): #サイノグラムの計算(画像データ, 直線方向分割数, 角度方向分割数, 相対誤差レベル)
  figuresize = img.shape[0]
  f = np.zeros( ( figuresize ** 2 , 1 ) )
  Ffigure = np.array(img)
  errorvec = np.zeros ( ( ( angle_division - 1 ) * ( s_division - 1 ) , 1) )
  errorvec_aux = np.zeros ( ( ( angle_division - 1 ) * ( s_division - 1 ) , 1) )
  B = calc_matrixA(img, angle_division, s_division)
  for i in range( 0 , figuresize ): # iはy方向
    for j in range( 0 , figuresize ): # j はx方向
      f[ j * figuresize + i ,0 ] = Ffigure[ i , j ]
  #
  #fnorm = np.linalg.norm( f )
  sinogram = B @ f # sinogramの計算
  sinogramnorm = np.linalg.norm( sinogram , ord =2 )
  for k in range(0 , ( angle_division - 1 ) * ( s_division - 1 ) ):
    errorvec_aux[ k , 0 ] = rd.uniform(-1,1)
  errorvec_aux_norm = np.linalg.norm( errorvec_aux , ord =2 )
  errorvec = ( sg_error * sinogramnorm / errorvec_aux_norm ) * errorvec_aux 
  sinogram_error = sinogram + errorvec # 誤差付きsiongramの計算
  #imgtest = np.zeros( ( figuresize , figuresize ) )
  #for k in range( 0 , figuresize ): # test
  #  for l in range( 0 , figuresize ):
  #      imgtest[ k , l ] = f[ k * figuresize + l , 0 ]
  return sinogram_error
#
#####################################################
def tikhonov(img, angle_division, s_division, alpha, sg_error ): #逆問題を解く
  figuresize = img.shape[0]
  imgt = np.zeros( ( figuresize , figuresize ) )
  At = calc_matrixA(img, angle_division, s_division)
  mt = calc_vectorf(img, angle_division, s_division, sg_error)
  left = (At.T) @ At + alpha * np.identity( figuresize ** 2 )
  right = (At.T) @ mt
  ft = np.linalg.solve( left , right )
  #ft = np.linalg.inv(( At.T @ At + alpha * np.identity( figuresize ** 2 ) )) @ (At.T) @ mt
  for i in range( 0 , figuresize ): # i はy方向
    for j in range( 0 , figuresize ): # j はx方向
      imgt[ i , j ] = ft[ j * figuresize + i , 0 ]
  return imgt
#####################################################
#
aa = np.array(Image.open('shepploganphantom128.png').convert('L'))
bb = 80 # 角分割数
cc = 120 # 位置分割数
dd = 100.00 # 正則化パラメータ
rerror = 0.01 #誤差レベル
reconstruct = tikhonov(aa,bb, cc, dd)
pil_img = Image.fromarray(reconstruct.astype(np.uint8)) #画像として出力
pil_img.save('reco_shepplogan_80_120_10000_er005.png') #画像をpngファイルで保存, ファイル名は適宜変更