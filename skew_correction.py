import cv2
import numpy as np

# read and scale down image
# wget https://bigsnarf.files.wordpress.com/2017/05/hammer.png #black and white
# wget https://i1.wp.com/images.hgmsites.net/hug/2011-volvo-s60_100323431_h.jpg
img = cv2.pyrDown(cv2.imread('miring.jpg', cv2.IMREAD_UNCHANGED))

# threshold image
ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                  127, 255, cv2.THRESH_BINARY)
# find contours and get the external one

contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE,
#                cv2.CHAIN_APPROX_SIMPLE)

# with each contour, draw boundingRect in green
# a minAreaRect in red and
# a minEnclosingCircle in blue
w_angles = np.array([])
h_angles = np.array([])
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # get the min area rect
    rect = cv2.minAreaRect(c)
    # print(rect)
    center, hw, angle = rect
    hi, wd = hw
    # angle = cv2.minAreaRect(c)[-1]
    # if angle < -45:
    angle = (90 + angle)
    # else:
    #     angle = -angle
    if angle != 0.0:
        if wd > hi:
            w_angles = np.append(w_angles, [angle])
        else:
            h_angles = np.append(h_angles, [angle])
    # print(angle)
    box = cv2.boxPoints(rect)
    # print(box)
    # convert all coordinates floating point values to int
    box = np.int0(box)
    # draw a red 'nghien' rectangle
    cv2.drawContours(img, [box], 0, (0, 0, 255))

    # finally, get the min enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(c)
    # convert all values to int
    center = (int(x), int(y))
    radius = int(radius)
    # and draw the circle in blue
    img = cv2.circle(img, center, radius, (255, 0, 0), 2)

# print(len(contours))
cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

# cv2.imshow("contours", img)

(h, w) = threshed_img.shape[:2]
center = (w // 2, h // 2)
a = 0
print(w_angles)
print(h_angles)
if h_angles.size > w_angles.size:
    a = np.mean(h_angles)
else:
    a = np.mean(w_angles)
print(a)
M = cv2.getRotationMatrix2D(center, a, 1.0)
rotated = cv2.warpAffine(threshed_img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REPLICATE)

cv2.imshow("rotated", rotated)

cv2.waitKey(0)
# while True:
#     key = cv2.waitKey(1)
#     if key == 27:  # ESC key to break
#         break

# cv2.destroyAllWindows()
