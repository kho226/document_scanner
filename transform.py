import cv2
import numpy as np

def orderPoints(pts): #pts is a list of four points specifying the edges of a rectangle
    #initialize a list of coordinates
    #ordered from top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4,2), dtype = "float32")
    
    #the top-left will have the smallest sum of (x,y)
    #the bottom-right will have the largest sum of (x,y)
    s = pts.sum(axis = 1) #sum the coordinates
    rect[0] = pts[np.argmin(s)] #top-left
    rect[2] = pts[np.argmax(s)] #top-right

    #compute the difference between the points
    #top-right will have the smallest difference
    #bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def fourPointTransform(image, pts): #transform the coordinates of image to coordinates in pts
    rect = orderPoints(pts)
    (topL, topR, bottomR, bottomL) = rect #unpack the points individually

    #find the width of the new image
    widthA = np.sqrt(((topL[0] - topR[0])**2) + ((topL[1] - bottomR[1])**2))
    widthB = np.sqrt(((bottomL[0] - bottomR[0])**2) + ((bottomL[1] - bottomR[1])**2))
    newWidth = max(int(widthA), int(widthB))

    #find the height of the new image
    heightA = np.sqrt(((topL[0] - bottomL[0])**2) + ((topL[1] - bottomL[1])**2))
    heightB = np.sqrt(((topR[0] - bottomR[0])**2) + ((topR[1] - bottomR[1])**2))
    newHeight = max(int(heightA), int(heightB))

    #construct the set of destination points to obtain a top-down view
    #top-left, top-right, bottom-right, bottom-left
    dst = np.array ([
                    [0,0],
                    [newWidth-1, 0],
                    [newWidth-1, newHeight - 1],
                    [0, newHeight-1]],
                    dtype = "float32")
    #compute perspective transform
    #apply it
    M = cv2.getPerspectiveTransform(rect, dst) #matrix M
    warped = cv2.warpPerspective(image, M, (newWidth, newHeight)) #top-down view

    return warped
    
