import cvzone
import cv2
import numpy as np
import math
# hand tracking module
from cvzone.HandTrackingModule import HandDetector
import random


cap = cv2.VideoCapture(1)
# Width
cap.set(3, 1280)
#  Height
cap.set(4, 520)

detector = HandDetector(detectionCon=0.8, maxHands=1) # just use 1 hand

# What we need
# 1. List of points
# 2. List of distances
# 3. Current Length
# 4. Total Length

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = [] # 1
        self.lengths = [] # 2
        self.currentLength = 0 # 3
        self.allowedLength = 150 # start length
        self.previousHead = 0, 0 # previous head point

        # foor image path,
        # set to UNCHANGED to remove background from png
        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        # dimension or location of food
        self.hFood, self.wFood, _ = self.imgFood.shape
        # food point, set random
        self.foodPoint = 0, 0
        self.score = 0
        #initialize random food
        self.randomFoodLocation()

    def randomFoodLocation(self):
        self.foodPoint = (random.randint(100, 1000), random.randint(100,600))

    # method for update
    def update(self, imgMain, currentHead):
        px, py = self.previousHead
        cx, cy = currentHead

        # store the current head point to points
        self.points.append([cx, cy])
        # append the distance
        distance = math.hypot(cx-px, cy-py)
        self.lengths.append(distance)
        # once we get the distance, add to the current length
        self.currentLength += distance
        # update the previous head
        self.previousHead = cx,cy

        # LENGTH REDUCTION
        # check if currentlength grater than allowed length, to make sure fixed length by reducing line
        if self.currentLength > self.allowedLength:
            for i, length in enumerate(self.lengths):
                self.currentLength -= length
                # reduce the points and the lengths
                self.lengths.pop(i)
                self.points.pop(i)

                if self.currentLength < self.allowedLength:
                    break

        if self.points:
            # DRAW SNAKE IF WE HAVE POINTS
            for i,point in enumerate(self.points):
                if i != 0:
                    # point line to the current point
                    cv2.line(imgMain, self.points[i-1], self.points[i], (0,0,255), 20)
                    cv2.line(imgMain, self.points[i - 1], self.points[i], (0, 255, 0), 5)
            # draw circle in finger
            cv2.circle(imgMain, self.points[-1], 20, (255,0,0), cv2.FILLED)
        # DRAW FOOD
        # imgMain : background img
        # self.imgFood
        rx, ry = self.foodPoint
        cv2.circle(imgMain, (rx, ry), 30, (255, 185, 0), cv2.FILLED)
        if abs(rx-cx) <= 15 and abs(ry-cy) <= 15:
            self.randomFoodLocation()
            self.allowedLength += 50
            self.score += 1
        cvzone.putTextRect(imgMain, f"Score: {self.score}", [50,80],
                           scale=3, thickness=3, offset=20)
        return imgMain

game = SnakeGameClass("mango.png")

while True:
    success,img = cap.read()
    #flip image, to make it easier for us
    img = cv2.flip(img, 1)
    # find hand
    hands, img = detector.findHands(img, flipType=False)

    # find the point of hand in finger
    if hands:
        # landmark list
        lmList = hands[0]['lmList']
        # current head position, which is our finger
        pointIndex = lmList[8][0:2] # we wouldnt need 3
        # update image from class, set to img to be shown to the cap
        img = game.update(img, pointIndex)

    cv2.imshow('Image', img)
    cv2.waitKey(1)

