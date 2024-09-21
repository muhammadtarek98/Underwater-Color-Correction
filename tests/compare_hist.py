import cv2
import numpy as np
from scipy.signal.signaltools import signaltonoise
import math

def gdl(generated, true):

   true_x_shifted_right = true[1:,:,:]
   true_x_shifted_left = true[:-1,:,:]
   true_x_gradient = np.abs(true_x_shifted_right - true_x_shifted_left)

   generated_x_shifted_right = generated[1:,:,:]
   generated_x_shifted_left = generated[:-1,:,:]
   generated_x_gradient = np.abs(generated_x_shifted_right - generated_x_shifted_left)
   
   loss_x_gradient = np.linalg.norm(true_x_gradient - generated_x_gradient)

   true_y_shifted_right = true[:,1:,:]
   true_y_shifted_left = true[:,:-1,:]
   true_y_gradient = np.abs(true_y_shifted_right - true_y_shifted_left)

   generated_y_shifted_right = generated[:,1:,:]
   generated_y_shifted_left = generated[:,:-1,:]
   generated_y_gradient = np.abs(generated_y_shifted_right - generated_y_shifted_left)
    
   loss_y_gradient = np.linalg.norm(true_y_gradient - generated_y_gradient)

   loss = loss_x_gradient + loss_y_gradient

   return loss

def psnr(img1, img2):
   mse = np.mean( (img1 - img2) ** 2 )
   return 10 * math.log10(255.0**2/mse)

for i in range(1,5):
   flickr = cv2.imread('result_images/flickr_crop'+str(i)+'_og.png')
   cgan   = cv2.imread('result_images/cgan_crop'  +str(i)+'_og.png')
   ugan   = cv2.imread('result_images/ugan_crop'  +str(i)+'_og.png')
   uganp  = cv2.imread('result_images/uganp_crop' +str(i)+'_og.png')
   flickr_hist = cv2.calcHist([flickr], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   cgan_hist   = cv2.calcHist([cgan],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   ugan_hist   = cv2.calcHist([ugan],   [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   uganp_hist  = cv2.calcHist([uganp],  [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
   a = cv2.compareHist(flickr_hist, cgan_hist,  cv2.HISTCMP_BHATTACHARYYA)
   b = cv2.compareHist(flickr_hist, ugan_hist,  cv2.HISTCMP_BHATTACHARYYA)
   c = cv2.compareHist(flickr_hist, uganp_hist, cv2.HISTCMP_BHATTACHARYYA)
   flickr = flickr/255.0
   cgan   = cgan/255.0
   ugan   = ugan/255.0
   uganp  = uganp/255.0
   print ('& ' + str(round(np.mean(flickr), 2)) + ' $\pm$ ' + str(round(np.std(flickr), 2)) + ' & ' + str(round(np.mean(cgan), 2)) + ' $\pm$ ' + str(round(np.std(cgan), 2)) + ' & ' + str(round(np.mean(ugan), 2)) + ' $\pm$ ' + str(round(np.std(ugan), 2)) + ' & ' + str(round(np.mean(uganp), 2)) + ' $\pm$ ' + str(round(np.std(uganp), 2)) + ' \\\ \hline')

