
#a document scanner application using Python, OpenCV
from transform import fourPointTransform
from skimage.filters import threshold_adaptive
import numpy as np
import argparse
import cv2

def resizeImageHeight(image, height):
    r =  float(height)/image.shape[0] #image.shape -> (height, width)
    dim = (int(image.shape[1] * r), height) #(width,height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized

#construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help = "Path to the image to be scanned")
args = vars(ap.parse_args())


#load image off of the disk
image = cv2.imread(args["image"])
#resize the scanned image to a height of 500
ratio = image.shape[0]/500.0
orig = image.copy()
#dim = (int(image.shape[1] * r), 500)
image = resizeImageHeight(image, 500)

#convert the image to grayscale, blur it, and find edges in the image
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)

#show the original image and edge detected image
print "Edge Detection"
cv2.imshow("Image",image)
cv2.imshow ("Edged",edged)
print "Hit 0 to Find Contours of the Document"
cv2.waitKey(0)
cv2.destroyAllWindows()



# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
 
# show the contour (outline) of the piece of paper
print "Contours"
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
print "Hit 0 to Apply Perspective Transform"
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = fourPointTransform(orig, screenCnt.reshape(4,2) * ratio) #perform scan on the original
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
warped = threshold_adaptive(warped, 250, offset = 10)
warped = warped.astype("uint8") * 255

print "Perspective transform"
cv2.imshow("Original", resizeImageHeight(orig,650))
cv2.imshow("Scanned", resizeImageHeight(warped,650))
print "Hit 0 to quit the program"
cv2.waitKey(0)




