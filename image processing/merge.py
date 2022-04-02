from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from datetime import date , datetime
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments using left-most object as a reference(normally width = 0.25)
# -i is the image for the program
# -w is the width of the left-most reference object in c.m.
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in centimeter)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

#both detect and size
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

#hsv colors convert
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Light Yellow Color
low_ly = np.array([26, 36, 127]) # 0,38,0
high_ly = np.array([179, 255, 255]) # 44,255,255
ly_mask = cv2.inRange(hsv_image, low_ly, high_ly)
ly = cv2.bitwise_and(image, hsv_image, mask=ly_mask)
new_ly = ly.copy()
# Contours for ly
ly_grey = cv2.cvtColor(new_ly, cv2.COLOR_BGR2GRAY)
ly_grey = cv2.GaussianBlur(new_ly, (7, 7), 0)
edged_ly = cv2.Canny(ly_grey, 30, 200)
contoursly, hierarchy = cv2.findContours(edged_ly, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

edged_ly = cv2.dilate(edged_ly, None, iterations=1)
edged_ly = cv2.erode(edged_ly, None, iterations=1)
# find contours in the edge map
cnts_ly = cv2.findContours(edged_ly.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_ly = imutils.grab_contours(cnts_ly)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts_ly, _) = contours.sort_contours(cnts_ly)

cv2.drawContours(new_ly, contoursly, -1, (0, 255, 0), 3)
# cv2.rectangle(new_ly)
cv2.namedWindow('Contours Yellow', cv2.WINDOW_NORMAL)
cv2.imshow('Contours Yellow', new_ly)
cv2.resizeWindow('Contours Yellow', 600, 600)
cv2.waitKey(0)

# Yellow 1
low_ly_1 = np.array([24,20,189])
high_ly_1 = np.array([179, 255, 255])
ly_1_mask = cv2.inRange(hsv_image, low_ly_1, high_ly_1)
ly_1 = cv2.bitwise_and(image, hsv_image, mask=ly_1_mask)
new_ly_1 = ly_1.copy()
# Contours for yl_1
ly_1_grey = cv2.cvtColor(new_ly_1, cv2.COLOR_BGR2GRAY)
ly_1_grey = cv2.GaussianBlur(new_ly_1, (7, 7), 0)
edged_ly1 = cv2.Canny(ly_1_grey, 30, 200)
contoursly, hierarchy = cv2.findContours(edged_ly1, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
edged_ly1 = cv2.dilate(edged_ly1, None, iterations=1)
edged_ly1 = cv2.erode(edged_ly1, None, iterations=1)
# find contours in the edge map
cnts_ly1 = cv2.findContours(edged_ly1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_ly1 = imutils.grab_contours(cnts_ly1)
cv2.drawContours(new_ly_1, contoursly, -1, (0, 255, 0), 3)
cv2.namedWindow('Contours Yellow rice 1', cv2.WINDOW_NORMAL)
cv2.imshow('Contours Yellow rice 1', new_ly_1)
cv2.resizeWindow('Contours Yellow rice 1', 600, 600)
cv2.waitKey(0)

# White 1
low_wr_1 = np.array([18,28,186])
high_wr_1 = np.array([179, 255, 255])
wr_1_mask = cv2.inRange(hsv_image, low_wr_1, high_wr_1)
wr_1 = cv2.bitwise_and(image, hsv_image, mask=wr_1_mask)
new_wr_1 = wr_1.copy()
# Contours for wr_1
wr_1_grey = cv2.cvtColor(new_wr_1, cv2.COLOR_BGR2GRAY)
wr_1_grey = cv2.GaussianBlur(new_wr_1, (7, 7), 0)
edged_wr1 = cv2.Canny(wr_1_grey, 30, 200)
contoursly, hierarchy = cv2.findContours(edged_wr1, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
edged_wr1 = cv2.dilate(edged_wr1, None, iterations=1)
edged_wr1 = cv2.erode(edged_wr1, None, iterations=1)
# find contours in the edge map
cnts_wr1 = cv2.findContours(edged_wr1.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_wr1 = imutils.grab_contours(cnts_wr1)
cv2.drawContours(new_wr_1, contoursly, -1, (0, 255, 0), 3)
cv2.namedWindow('Contours White rice 1', cv2.WINDOW_NORMAL)
cv2.imshow('Contours White rice 1', new_wr_1)
cv2.resizeWindow('Contours White rice 1', 600, 600)
cv2.waitKey(0)

# White rice Color
low_wr = np.array([38, 4, 209]) # 37,0,203
high_wr = np.array([59,36,255]) # 131,255,255
wr_mask = cv2.inRange(hsv_image, low_wr, high_wr)
wr = cv2.bitwise_and(image, hsv_image, mask=wr_mask)
new_wr = wr.copy()
# Contours for wr
wr_grey = cv2.cvtColor(new_wr, cv2.COLOR_BGR2GRAY)
wr_grey = cv2.GaussianBlur(new_wr, (7, 7), 0)
edged_wr = cv2.Canny(wr_grey, 30, 200)
contoursly, hierarchy = cv2.findContours(edged_wr, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
edged_wr = cv2.dilate(edged_wr, None, iterations=1)
edged_wr = cv2.erode(edged_wr, None, iterations=1)
# find contours in the edge map
cnts_wr = cv2.findContours(edged_wr.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts_wr = imutils.grab_contours(cnts_wr)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
# (cnts_wr, _) = contours.sort_contours(cnts_wr)

cv2.drawContours(new_wr, contoursly, -1, (0, 255, 0), 3)
cv2.namedWindow('Contours White rice', cv2.WINDOW_NORMAL)
cv2.imshow('Contours White rice', new_wr)
cv2.resizeWindow('Contours White rice', 600, 600)
cv2.waitKey(0)

# cv2.namedWindow('result for yellow', cv2.WINDOW_NORMAL)
# cv2.imshow( "result for yellow", ly)
# cv2.resizeWindow('result for yellow', 600, 600)
# cv2.waitKey(0)
# cv2.namedWindow('result for yellow 1', cv2.WINDOW_NORMAL)
# cv2.imshow( "result for yellow 1", ly_1)
# cv2.resizeWindow('result for yellow 1', 600, 600)
# cv2.waitKey(0)
# cv2.namedWindow('result for white', cv2.WINDOW_NORMAL)
# cv2.imshow( "result for white", wr)
# cv2.resizeWindow('result for white', 600, 600)
# cv2.waitKey(0)
# cv2.namedWindow('result for white 1', cv2.WINDOW_NORMAL)
# cv2.imshow( "result for white 1", wr_1)
# cv2.resizeWindow('result for white 1', 600, 600)
# cv2.waitKey(0)


# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

lightYellow_counts = 0
for c in cnts_ly:
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# put text on the image wheter it's white or yellow
	cv2.putText(orig, "Yellow",
		(int(trbrX + 10), int(trbrY + 50)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	lightYellow_counts += 1

	# show the output image
	cv2.imshow("Image with Yellow rice", orig)
	cv2.waitKey(0)
	
lightYellow_alternative_counts = 0
for c in cnts_ly1:
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# put text on the image wheter it's white or yellow
	cv2.putText(orig, "Yellow Alternative",
		(int(trbrX + 10), int(trbrY + 50)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	lightYellow_alternative_counts += 1

	# show the output image
	cv2.imshow("Image with Yellow rice alternative", orig)
	cv2.waitKey(0)

white_alternative_count = 0
for c in cnts_wr1:
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# put text on the image wheter it's white or yellow
	cv2.putText(orig, "White Alternative",
		(int(trbrX + 10), int(trbrY + 50)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	white_alternative_count += 1

	# show the output image
	cv2.imshow("Image with white rice alternative", orig)
	cv2.waitKey(0)

white_count = 0
for c in cnts_wr:
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# put text on the image wheter it's white or yellow

	# if dimA & dimB <= "2.20":
	# 	cv2.putText(orig, "White",
	# 	(int(trbrX + 10), int(trbrY + 50)), cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.65, (255, 255, 255), 2)
	# else:
	# 	cv2.putText(orig, "Reference object",
	# 	(int(trbrX + 10), int(trbrY + 50)), cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.65, (255, 255, 255), 2)

	cv2.putText(orig, "White",
		(int(trbrX + 10), int(trbrY + 50)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

	white_count += 1

	# show the output image
	cv2.imshow("Image with white rice", orig)
	cv2.waitKey(0)

# loop over the contours individually
referenceObject = 0
first_array = []
second_array = []
array_stack = []
rice_count = 0
# tltrX tltrY blbrX blbrY , tlblX tlblY trbrX trbrY
tltrX_array = []
tltrY_array = []
blbrX_array = []
blbrY_array = []
tlblX_array = []
tlblY_array = []
trbrX_array = []
trbrY_array = []
box_array = []
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 100:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)
	box_array.append(box)
	cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
	# loop over the original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
    # unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	# compute the midpoint between the top-left and top-right points,
	# followed by the midpoint between the top-righ and bottom-right
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	# append values to array of each points
	tltrX_array.append(tltrX)
	tltrY_array.append(tltrY)
	blbrX_array.append(blbrX)
	blbrY_array.append(blbrY)
	tlblX_array.append(tlblX)
	tlblY_array.append(tlblY)
	trbrX_array.append(trbrX)
	trbrY_array.append(trbrY)

    # compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	# if the pixels per metric has not been initialized, then
	# compute it as the ratio of pixels to supplied metric
	# (in this case, centimeter)
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]
        
    # compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	if dimA >= 2.20:
		referenceObject += 1

	# draw the object sizes on the image
	# cv2.putText(orig, "{:.1f}cm".format(dimA),
	# 	(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.65, (255, 255, 255), 2)
	# cv2.putText(orig, "{:.1f}cm".format(dimB),
	# 	(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
	# 	0.65, (255, 255, 255), 2)
	np.array(first_array.append("{:.1f}".format(dimA)))
	np.array(second_array.append("{:.1f}".format(dimB)))
	stack = np.stack((first_array,second_array), axis=1)

	# rice counts
	rice_count += 1

# draw box of information
# height, width, color = image.shape
# imageWithBox = cv2.rectangle(orig, (1,int(height/2 + 350)), (int(width - 1),int(height - 1)) , (219, 104, 235), 2)

# draw lines between the midpoints on each objects
# draw the object sizes on the image
riceNum = 1
for i in range(len(tltrX_array)):
	posX = 10
	cv2.line(orig, (int(tltrX_array[i]), int(tltrY_array[i])), (int(blbrX_array[i]), int(blbrY_array[i])),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX_array[i]), int(tlblY_array[i])), (int(trbrX_array[i]), int(trbrY_array[i])),
		(255, 0, 255), 2)
	cv2.putText(orig, "{:.1f}cm".format(float(first_array[i])),
		(int(int(tltrX_array[i]) - 15), int(int(tltrY_array[i]) - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}cm".format(float(second_array[i])),
		(int(int(tlblX_array[i]) + 10), int(int(tlblY_array[i]))), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
	cv2.drawContours(orig, [box_array[i].astype("int")], -1, (0, 255, 0), 2)
	# cv2.putText(orig, "{}".format(riceNum),(int(int(tltrX_array[i]) - 15), int(int(tltrY_array[i]) - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
	# cv2.drawContours(orig, [box_array[i].astype("int")], -1, (0, 255, 0), 2)
	# cv2.putText(orig, "rice{}".format(riceNum),(posX, int(height/2 + 370)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
	
	posX += 10
	riceNum += 1

# show the output image
cv2.imshow("Image", orig)
cv2.waitKey(0)	

# get current date and time to named folder
today = datetime.now()
date_formatted = today.strftime("%d-%m-%Y %H:%M:%S")

file_name = "{}-{}.txt".format(rice_count, date_formatted)

with open(file_name, "w+") as f:
	data = f.read()
	f.write(str(stack) + "\n")
	f.write('Reference count(s) = {}, \n'.format(referenceObject))
	f.write('rice count = {}, \n'.format(rice_count - referenceObject))
	f.write('date = {} in format DD/MM/YYYY HH:MM:SS, \n'.format(date_formatted))
	f.write('image = {}, \n'.format(args["image"]))
	f.write('white count(s) = {}, \n'.format(white_count))
	f.write('white alternative count(s) = {}, \n'.format(white_alternative_count))
	f.write('yellow count(s) = {}, \n'.format(lightYellow_counts))
	f.write('yellow alternative count(s) = {}, \n'.format(lightYellow_alternative_counts))