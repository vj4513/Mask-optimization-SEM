from PIL import Image
import numpy as np
import cv2

# Binarise, filter and save
k=7
l_thres = 163
h_thres = 255
i = 150
j = 150

im = cv2.imread('0WOMem_G1_0r0.tif',2)
ret, im = cv2.threshold(im,l_thres,h_thres,cv2.THRESH_BINARY)
im = cv2.medianBlur(im, k)
#im = cv2.bitwise_not(im)
img = np.array(im,'int64')
im = Image.fromarray(im)
im.save('SEM.jpg')

# Crop image manually and convert MASK to jpg
#%% Resize images 

old_im1 = Image.open('SEM.jpg')
old_size1 = old_im1.size
old_im2 = Image.open('MASK.jpg')
old_size2 = old_im2.size

re_factor = (round(old_size2[0]/old_size1[0]*old_size1[0]),
             round(old_size2[0]/old_size1[0]*old_size1[1]))
old_im1 = old_im1.resize(re_factor)

k1 = tuple(round(i/2) for i in old_im1.size)
k2 = tuple(round(i/2) for i in old_im2.size)
old_im1 = old_im1.resize(k1)
old_im2 = old_im2.resize(k2)
old_im1.save('SEM_b.jpg')
old_im2.save('MASK_b.jpg')

#%% Align the two images

im1 =  cv2.imread("MASK_b.jpg")
im2 =  cv2.imread("SEM_b.jpg")
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

sz = im1.shape
warp_mode = cv2.MOTION_AFFINE

if warp_mode == cv2.MOTION_HOMOGRAPHY :
    warp_matrix = np.eye(3, 3, dtype=np.float32)
else :
    warp_matrix = np.eye(2, 3, dtype=np.float32)

number_of_iterations = 5000;
termination_eps = 1e-10;

criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
            number_of_iterations,  termination_eps)
(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,
        warp_matrix, warp_mode, criteria)

if warp_mode == cv2.MOTION_HOMOGRAPHY :
    im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else :
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

cv2.imwrite('Aff_align_p4.jpg',im2_aligned)

#%% Filter, compare and mark the areas

im1 = Image.open('Aff_align_p4.jpg').convert('L')
im2 = Image.open('MASK_b.jpg').convert('L')
#im1 = cv2.medianBlur(im1, 9)
im1b = np.array(im1,'int64')
im2b = np.array(im2,'int64')
im1b[im1b < 100] = 0
im1b[im1b >= 100] = 1
im2b[im2b < 100] = 0
im2b[im2b >= 100] = 1

diff = im1b - im2b
img=cv2.imread('MASK_b.jpg')
r_count = 0
g_count = 0

for i in range(len(diff)):
    for j in range(len(diff[0])):
         if i > 150 and i < (len(diff) - 150) and j  > 150 and j < (len(diff[0]) - 150):
             if diff[i,j] >= 1:
                img[i,j,0] = 255
                img[i,j,1] = 0
                img[i,j,2] = 0
                r_count = r_count + 1
             elif diff[i,j] <= -1:
                img[i,j,0] = 0
                img[i,j,1] = 255
                img[i,j,2] = 0 
                g_count = r_count + 1

im = Image.fromarray(img)
im.save('p4.jpg')
