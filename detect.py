import cv2
import numpy as np
from scipy.ndimage import interpolation as inter
import math


image = cv2.pyrDown(cv2.imread('miring.jpg', cv2.IMREAD_UNCHANGED))


def preprocessing(img):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # brightnes-contrast-noise reduction
    brightness = 50
    contrast = 30
    brighten = np.int16(gray)
    brighten = brighten * (contrast/127+1) - contrast + brightness
    brighten = np.clip(brighten, 0, 255)
    brighten = np.uint8(brighten)
    denoise = cv2.fastNlMeansDenoising(brighten)
    # biner
    ret, thresh = cv2.threshold(denoise, 127, 255, cv2.THRESH_BINARY)
    # skew correction
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    w_angles = np.array([])
    h_angles = np.array([])
    for c in contours:
        rect = cv2.minAreaRect(c)
        center, hw, angle = rect
        hi, wd = hw
        angle = (90 + angle)
        if angle != 0.0:
            if wd > hi:
                w_angles = np.append(w_angles, [angle])
            else:
                h_angles = np.append(h_angles, [angle])
        box = cv2.boxPoints(rect)
        # print(box)
        # convert all coordinates floating point values to int
        box = np.int0(box)
        # draw a red 'nghien' rectangle
        cv2.drawContours(brighten, [box], 0, (0, 0, 255))

    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    a = 0
    if h_angles.size > w_angles.size:
        a = np.mean(h_angles)
    else:
        a = np.mean(w_angles)
    M = cv2.getRotationMatrix2D(center, a, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    # resize
    res_h = int(img.shape[0] * .8)
    res_w = int(img.shape[1] * .8)
    resized = cv2.resize(rotated, (res_w, res_h),  interpolation=cv2.INTER_AREA)
    # thinning
    invert = cv2.bitwise_not(resized)
    thinned = cv2.ximgproc.thinning(invert)
    # segmentasi (connected component labelling)
    connectivity = 8
    ccl = cv2.connectedComponentsWithStats(thinned, connectivity=connectivity)
    num_labels = ccl[0]
    labels_im = np.copy(ccl[1])
    stats = ccl[2]
    stats = np.array(sorted(stats, key=lambda x: x[cv2.CC_STAT_LEFT]))
    centroid = ccl[3]
    # print(labels_im)
    # print(np.array(labels_im))
    cv2.imshow('labelled', labels_im)
    label_hue = np.uint8(179*labels_im/np.max(labels_im))
    label_hue = np.uint8(labels_im)
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue != 0] = 0
    labeled_img[label_hue == 0] = 255
    boxed = np.copy(labeled_img)
    for i in range(num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if x == 0:
            continue
        cv2.rectangle(boxed, (x, y), (x+w, y+h), (0, 0, 0), 1)
        print("{} - {} | {} - {}".format(x, x+w, y, y+h))
    # print(stats)
    return labeled_img, boxed, num_labels, stats


img, labelled, num, stats = preprocessing(image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(img)

# cv2.imshow('preprocessed', img)
roi = []
for i in range(num):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    if x == 0:
        continue
    roi = img[y:y+h, x:x+w]
    # print('roi')
    # print(roi)
    # cv2.imshow('roi_{}'.format(i), roi)
# cv2.imshow('original image', img)
# cv2.imshow('#1 grayscale', gray)
# cv2.imshow('#2 brightnes-contrast-noise reduction', denoise)
# cv2.imshow('#3 binary tresholding', thresh)
# cv2.imshow('#4 skew correction', rotated)
# cv2.imshow('#5 resize', resized)
# cv2.imshow('#6 thinning', thinned)
# cv2.imshow('#7 segmentasi (connected component labelling)', labeled_img)
cv2.waitKey(0)
