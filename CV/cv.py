import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Resizing the frame size
def rescaleframe(frame, scale=.7):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA) # Area - big to small, Cubic - small to big image resizing


# Reading images
img = cv.imread("E:\ComputerVision\Dog_image.jpg")
# cv.imshow('Image',img)

# Blank image
blank = np.zeros(img.shape[:2], dtype = 'uint8')
blank1 = np.zeros((300,300), dtype='uint8')

# Bitwise operation
rec = cv.rectangle(img=blank1.copy(), pt1=(30,30), pt2=(200,200), color=255, thickness=-1)
cir = cv.circle(img=blank1.copy(), center=(150,150), radius=100, color=255, thickness=-1)
# cv.imshow('Rectangle', rec)
# cv.imshow('Circle', cir)

# Bitwise AND operation
bitand = cv.bitwise_and(rec, cir)
# cv.imshow('BitAND', bitand)

# Bitwise OR operation
bitor = cv.bitwise_or(rec, cir)
# cv.imshow('BitOR', bitor)

# Bitwise XOR operation
bitxor = cv.bitwise_xor(rec, cir)
# cv.imshow('BitXOR', bitxor)

# Bitwise NOT operation
bitnot = cv.bitwise_not(rec)
# cv.imshow('BitNOT', bitnot)

# Color Scales 
# Gray images
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('GrayImage', gray)

# Gray to HSV is not possible.

# HSV images
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# cv.imshow('HSVImage', hsv)

# LAB images
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
# cv.imshow('LABImage', lab)

# RGB format
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# cv.imshow('RGBImage', rgb)

# plt.imshow(rgb)
# plt.show()

# Color channels
b,g,r = cv.split(img)
# cv.imshow('Blue', b)
# cv.imshow('Green', g)
# cv.imshow('Red', r)

merged = cv.merge([b,g,r])
# cv.imshow('MergedImage', merged)

mblue = cv.merge([b,blank,blank])
mgreen = cv.merge([blank,g,blank])
mred = cv.merge([blank,blank,r])
# cv.imshow('Blue', mblue)
# cv.imshow('Green', mgreen)
# cv.imshow('Red', mred)

# Blur images
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
average = cv.blur(img, (3,3))
median = cv.medianBlur(img, 3)
bilateral = cv.bilateralFilter(img, 10, 35, 25)
# cv.imshow('BlurImage', blur)

# Edge Cascade
canny = cv.Canny(img, 125, 125)
canny1 = cv.Canny(blur, 125, 125) # Reduces the edge by using blur image
# cv.imshow('Canny', canny) 
# cv.imshow('Canny1', canny1)

# Image Threshold
ret, thres = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
# cv.imshow('ThresImage', thres)

# Image Contours
contours, hierarchies = cv.findContours(canny1, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# print(len(contours))

cv.drawContours(blank, contours, -1, (145, 200, 54), 1)
# cv.imshow('Countour', blank)

# Image dilation
dilate = cv.dilate(canny, (3,3), iterations=5)
# cv.imshow('DilatedImage',dilate)

# Image Erosion
erode = cv.erode(dilate, (3,3), iterations=2)
# cv.imshow('ErodedImage', erode)

# Cropped Images
crop = img[20:300, 40:100]
# cv.imshow('CroppedImage', crop)

# Image Translation
def translate(img, x,y):
    transMat = np.float32([[1,0,x],[0,1,y]])
    dim = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dim)

# -x --> left, -y --> up, x --> right, y --> down

transImage = translate(img, 100,100)
# cv.imshow('TranslatedImage', transImage)

# Image Rotation
def rotate(img, angle, rotpoint=None):
    (height, width) = img.shape[:2]

    if rotpoint is None:
        rotpoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotpoint, angle, scale=1.0)
    dim = (width, height)
    return cv.warpAffine(img, rotMat, dim)

rotated = rotate(img, 30)
# cv.imshow('RotatedImage', rotated)

# Image Flipping
flip = cv.flip(img, -1) 
# cv.imshow('FlippedImage1', flip)

# Face Detection
img_s = cv.imread('SinglePerson.jpg')
img_m = cv.imread('MultiplePerson.jpg')

haar_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect_s = haar_cascade.detectMultiScale(image=img_s, scaleFactor=1.1, minNeighbors=2)
faces_rect_m = haar_cascade.detectMultiScale(image=img_m, scaleFactor=1.1, minNeighbors=11)

print(f'No of faces found = {len(faces_rect_m)}')

for (x,y,w,h) in faces_rect_m:
    cv.rectangle(img_m, (x,y), (x+w, y+h), (200,150,100), 2)

cv.imshow('DetectedFaces',img_m)

# Reading videos
capture = cv.VideoCapture("E:\ComputerVision\Cat_video.mp4")

while True:
    isTrue, frame = capture.read()
    newframe = rescaleframe(frame)

    if isTrue:
        # cv.imshow('Video',frame)
        # cv.imshow('RescaledVideo', newframe)
        pass
    else:
        break

    if cv.waitKey(20)& 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()