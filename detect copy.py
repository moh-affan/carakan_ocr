import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import math


# def correct_skew(image, delta=1, limit=5):
#     def determine_score(arr, angle):
#         data = inter.rotate(arr, angle, reshape=False, order=0)
#         histogram = np.sum(data, axis=1)
#         score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
#         return histogram, score

#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#     scores = []
#     angles = np.arange(-limit, limit + delta, delta)
#     for angle in angles:
#         histogram, score = determine_score(thresh, angle)
#         scores.append(score)

#     best_angle = angles[scores.index(max(scores))]

#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC,
#                              borderMode=cv2.BORDER_REPLICATE)

#     return best_angle, rotated


# def determine_score(arr, angle):
#     data = inter.rotate(arr, angle, reshape=False, order=0)
#     histogram = np.sum(data, axis=1)
#     score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
#     return histogram, score


img = cv2.imread('miring.jpg')
# 1) grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 2) brighten image
# brighten = gray + 50
brightness = 50
contrast = 30
brighten = np.int16(gray)
brighten = brighten * (contrast/127+1) - contrast + brightness
brighten = np.clip(brighten, 0, 255)
brighten = np.uint8(brighten)
denoise = cv2.fastNlMeansDenoising(brighten)
# 3) binary
ret, thresh = cv2.threshold(denoise, 127, 255, cv2.THRESH_BINARY)
# thresh = cv2.threshold(denoise, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# cv2.imshow('gray', gray)
# cv2.imshow('brightnes', brighten)
# print(img.shape[0]) #h
# print(img.shape[1]) #w
# 4) resize image
res_h = int(img.shape[0] * .5)
res_w = int(img.shape[1] * .5)
resized = cv2.resize(thresh, (res_w, res_h),  interpolation=cv2.INTER_AREA)
# 5) skew correction
# delta = 1
# limit = 90
# scores = []
# angles = np.arange(-limit, limit + delta, delta)
# print(angles)
# for angle in angles:
#     histogram, score = determine_score(thresh, angle)
#     scores.append(score)
# print(scores)
# best_angle = angles[scores.index(max(scores))]
# print(best_angle)

# (h, w) = resized.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
# rotated = cv2.warpAffine(resized, M, (w, h), flags=cv2.INTER_CUBIC,
#                          borderMode=cv2.BORDER_REPLICATE)
contours, hier = cv2.findContours(resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
bounding = resized.copy()
w_angles = np.array([])
h_angles = np.array([])
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(bounding, (x, y), (x+w, y+h), (0, 255, 0), 2)
    rect = cv2.minAreaRect(c)
    center, hw, angle = rect
    hi, wd = hw
    angle = (90 + angle)
    if angle != 0.0:
        if wd > hi:
            w_angles = np.append(w_angles, [angle])
        else:
            h_angles = np.append(h_angles, [angle])
(h, w) = resized.shape[:2]
center = (w // 2, h // 2)
a = 0
# print(w_angles)
# print(h_angles)
if h_angles.size > w_angles.size:
    a = np.mean(h_angles)
else:
    a = np.mean(w_angles)
print(a)
M = cv2.getRotationMatrix2D(center, a, 1.0)
rotated = cv2.warpAffine(resized, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)
# height, width = invert.shape[0:2]
# minLineLength = width/2.0
# maxLineGap = 20
# lines = cv2.HoughLinesP(invert, 1, np.pi/180, 2, minLineLength, maxLineGap)
# print(lines)
# # calculate the angle between each line and the horizontal line:
# angle = 0.0
# nb_lines = len(lines)


# for line in lines:
#     angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0, line[0][2]*1.0 - line[0][0]*1.0)

# angle /= nb_lines*1.0

# angle = angle * 180.0 / np.pi
# print(angle)

# non_zero_pixels = cv2.findNonZero(resized)
# center, wh, theta = cv2.minAreaRect(non_zero_pixels)

# root_mat = cv2.getRotationMatrix2D(center, angle, 1)
# rows, cols = resized.shape
# rotated = cv2.warpAffine(invert, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
# sizex = np.int0(wh[0])
# sizey = np.int0(wh[1])
# if theta > -45:
#     temp = sizex
#     sizex = sizey
#     sizey = temp
# rotated = cv2.getRectSubPix(rotated, (sizey, sizex), center)
# 6) thinning / skeleton (erode)
# # kernel = np.ones((3, 3), np.uint8)
# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# # erosion = cv2.erode(invert, kernel, iterations=1)
# cp_img = invert.copy()
# thin = np.zeros(cp_img.shape, dtype='uint8')
# while(cv2.countNonZero(cp_img) != 0):
#     erode = cv2.erode(cp_img, kernel)
#     opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
#     subset = erode - opening
#     thin = cv2.bitwise_or(subset, thin)
#     cp_img = erode.copy()
invert = cv2.bitwise_not(rotated)
thinned = cv2.ximgproc.thinning(invert)
# thinned = cv2.bitwise_not(thinned)
# thinned = cv2.fastNlMeansDenoising(thinned, h=10)
# X) skew correction
# coords = np.column_stack(np.where(invert > 0))
# print(coords)
# angle = cv2.minAreaRect(coords)[-1]
# print(angle)
# if angle < -45:
#     angle = -(90 + angle)
# else:
#     angle = -angle
# print(angle)
# (h, w) = resized.shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated = cv2.warpAffine(thinned, M, (w, h),
#                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
cv2.imshow('brighten', brighten)
# cv2.imshow('denoise', denoise)
# cv2.imshow('thres', thres)
# cv2.imshow('resized', resized)
cv2.imshow('bounding', bounding)
# cv2.imshow('invert', invert)
cv2.imshow('rotated', rotated)
cv2.imshow('thinning', thinned)
cv2.waitKey(0)
