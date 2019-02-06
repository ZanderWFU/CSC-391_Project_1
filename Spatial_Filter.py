import cv2
import numpy as np
import time

print("Enter the name of the file:")
read_image_name = input()

image = cv2.imread(read_image_name)





print("What type of blur will be applied to the image? (Box, Gaussian, Median, None)")
invalid = True
mode_dict = {"box": 'B', "b": 'B', "gaussian": 'G', "g": 'G', "median": 'M', "m": 'M', "none": 'N', "n": 'N' }
while (invalid):
    mode_input = input()
    mode_input = mode_input.lower()
    if mode_input in mode_dict:
        invalid = False
mode = mode_dict[mode_input]

print("Enter a value for k to be the size of the filter:")
k = int(input())  #k x k filter
if mode == 'G':
    print("Enter a value for sigma for use in the Gaussian filter:")
    sigma = int(input())

print("Run edge detection?")
invalid = True
edge_dict = {"yes": True, "y": True, "no": False, "n": False}
while (invalid):
    edge_input = input()
    edge_input = edge_input.lower()
    if edge_input in edge_dict:
        invalid = False
edge = edge_dict[edge_input]
#running canny edge detector
print("Running Now...")
if mode == 'M':
    print("The median blur takes a while")

low = 50
high = 240
save_image = True #Will the images be stored as .jpgs


type = 'NA' #Will be used for file naming
siginfo = ''  #optional part of file naming
start = time.time() #start the stopwatch
middle = int(k/2) #middle of the array, useful in formulas
num_rows, num_col = image.shape[:2] #Gets the dimensions of the original image

cv2.imshow("Original Image", image) #display the image

#Basic Box Filter (for testing purposes)
if (mode == 'B'):
    box = np.ones((k,k), np.float32) / (k*k) #creates the box filter
    filtered = cv2.filter2D(image, -1, box)
    type = 'Box'
    siginfo = f'{k}x{k}'

#Gaussian Filter
if (mode == 'G'):
    box = np.ones((k,k), np.float32)
    for x in range(k):
        for y in range(k):
            box[x,y] = 1 /(2 * np.pi * sigma ** 2) * np.e ** (-1 * ((x - middle) ** 2 + (y - middle) ** 2)/(2 * sigma ** 2))

    sum = np.sum(box)
    box = box / sum # This math makes it so that the box sums to 1, so the image doesn't darken or lighten

    filtered = cv2.filter2D(image, -1, box)
    type = 'Gau'
    siginfo = f'{k}x{k}s{sigma}'
    #print(box)

#Median Filter
if (mode == 'M'):
    #Bounds image for use with median filter
    bounding_matrix = np.float32(([1, 0, middle], [0,1, middle]))
    bounded = cv2.warpAffine(image, bounding_matrix, (num_col + 2*middle, num_rows+2*middle), cv2.INTER_LINEAR)
    filtered = np.ones(image.shape, np.uint8)
    for x in range(num_rows):
        for y in range(num_col):
            mat = bounded[x:x+2*middle, y:y+2*middle, 0]
            filtered[x,y, 0] = np.median(mat)
            mat = bounded[x:x + 2 * middle, y:y + 2 * middle, 1]
            filtered[x, y, 1] = np.median(mat)
            mat = bounded[x:x + 2 * middle, y:y + 2 * middle, 2]
            filtered[x, y, 2] = np.median(mat)
    type = 'Med'
    siginfo = f'{k}x{k}'

#No Filter
if (mode == 'N'):
    filtered = image

#Canny Edge Detection
if (edge):
    edge = cv2.Canny(filtered, low, high)
    cfilename = f'Canny_Edge_{type}_{low}_{high}.jpg'

    cv2.imshow('Edge Detection', edge)
    if save_image:
        cv2.imwrite(cfilename, edge)



cv2.imshow('Filtered Image', filtered)

print("Wait Time:", time.time() - start)
key = cv2.waitKey(0)
filename = f"{type}_{siginfo}.jpg"
if save_image:
    cv2.imwrite(filename, filtered)
cv2.destroyAllWindows()