
import numpy as np
import math
#import cv2

def calcBlockDesc(img, kp, left_top):
    imgRows = img.shape[0]
    imgCols = img.shape[1]

    mag_block = np.zeros((4,4), np.float)
    ori_block = np.zeros((4,4), np.int)

    main_angle = kp["angle"]
    ltx = left_top["x"]
    lty = left_top["y"]

    for y in range(lty, lty + 4):
        if y < 0 or y > imgRows - 2:
            continue

        for x in range(ltx, ltx + 4):
            if x < 0 or x > imgCols - 2:
                continue

            cur = np.float(img[y][x])
            right = np.float(img[y][x+1])
            bottom = np.float(img[y+1][x])

            mag_block[y - lty][x - ltx] = math.sqrt((right - cur) * (right - cur) + (bottom - cur) * (bottom - cur))
            #angle = math.atan2((bottom - cur), (right - cur)) - main_angle
            angle = cv2.fastAtan2(bottom - cur, right - cur) * math.pi / 180.0 - main_angle

            if angle < 0:
                angle = angle + 2 * math.pi
            ori_block[y - lty][x - ltx] = angle

    desc = countBlockMag(mag_block, ori_block, 4, 8)

    return desc

def calcPointDesc(img, kp):
    desc = []

    for ri in range(4):
        for ci in range(4):
            left_top_x = kp["x"] - 7 + ci * 4
            left_top_y = kp["y"] - 7 + ri * 4
            left_top = {"x": left_top_x,
                        "y": left_top_y}
            block = calcBlockDesc(img, kp, left_top)
            desc = np.hstack((desc, block))

    return desc

def calcPointSift(img, x, y):
    mag, ori = calcPointMag(img, x, y)

    kp = {"x": x, "y": y, "angle": ori}
    desc = calcPointDesc(img, kp)

    return mag, ori, desc


def countBlockMag(mag, ori, patch_size, div):
    magcounts = np.zeros(div, np.float)

    for i in range(patch_size):
        for k in range(patch_size):
            ind = np.int(div * ori[i][k] / (2 * math.pi))
            magcounts[ind] = magcounts[ind] + mag[i][k]

    return magcounts

def calcPointMag(img, x, y, patch_size = 5):
    mag_kp = np.zeros((patch_size, patch_size), dtype = np.float)
    ori_kp = np.zeros((patch_size, patch_size), dtype = np.float)

    imgRows = img.shape[0]
    imgCols = img.shape[1]

    shift = np.int((patch_size - 1) / 2)
    for r in range(patch_size):
        imgR = r - shift + y
        if imgR < 0 or imgR > imgRows - shift:
            continue

        for c in range(patch_size):
            imgC = c - shift + x
            if imgC < 0 or imgC > imgCols - shift:
                continue

            cur = np.float(img[imgR][imgC])
            right = np.float(img[imgR][imgC + 1])
            bottom = np.float(img[imgR + 1][imgC])
            mag_kp[r][c] = math.sqrt((right - cur) * (right - cur) + (bottom - cur) * (bottom - cur))
            #ori_kp[r][c] = math.atan2(bottom - cur, right - cur)
            ori_kp[r][c] = cv2.fastAtan2(bottom - cur, right - cur) * math.pi / 180.0

            if ori_kp[r][c] < 0:
                ori_kp[r][c] = ori_kp[r][c] + 2 * math.pi

    # count the maximum orientation and magnitude of the keypoint
    maxvm = 0.0
    maxvp = 0
    magcounts = countBlockMag(mag_kp, ori_kp, patch_size, 36)
    for k in range(36):
        if magcounts[k] > maxvm:
            maxvm = magcounts[k]
            maxvp = k

    #ori = np.int((maxvp + 0.5) * 10)
    ori = ((maxvp + 0.5) * 10.0) * math.pi / 180.0
    mag = maxvm

    return mag, ori
