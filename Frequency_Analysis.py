import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from mpl_toolkits.mplot3d import Axes3D

print("Enter the name of the file:")
image_name = input()


invalid = True
yn_dict = {"yes": True, "y": True, "no": False, "n": False}
dim_dict = {"2": 2, "two": 2, "2d": 2, "2-d": 2,"3": 3, "three": 3, "3d": 3, "3-d": 3}
box_dict = {"center": 'C', "c": 'C', "edge": 'E', "edges": 'E', "e": 'E'}

print("Plot in 2D or 3D?")
while (invalid):
    usr_input = input()
    usr_input = usr_input.lower()
    if usr_input in dim_dict:
        invalid = False
Dimension = dim_dict[usr_input]

print("Plot the log of the magnitude + 1?")
invalid = True
while (invalid):
    usr_input = input()
    usr_input = usr_input.lower()
    if usr_input in yn_dict:
        invalid = False
LogMag = yn_dict[usr_input]
print("Should a box be zero-ed on the DFT?")
invalid = True
while (invalid):
    usr_input = input()
    usr_input = usr_input.lower()
    if usr_input in yn_dict:
        invalid = False
zeroing = yn_dict[usr_input]
if zeroing:
    print("Where should the box(es) be? (Center, Edges)")
    invalid = True
    while (invalid):
        usr_input = input()
        usr_input = usr_input.lower()
        if usr_input in box_dict:
            invalid = False
    zeroing_mode = box_dict[usr_input]
    e = ''
    if zeroing_mode == 'E':
        e = 'es'
    print(f"How big should the box{e} be? (k x k)")
    zero_num = int(input())
    if zeroing_mode == 'E':
        zero_num = int(zero_num/2)

#settings:
color = True #if the image is in color it will have to be converted to grayscale
#Dimension = 2 #Display in 2-D or 3-D
#LogMag = True #if the log of the magnitude is being plotted for better visualization
saveplot = True #Should it save the 2-D image
savegray = False #Should it save the gray conversion of the image

#zeroing = False #zeroing part of the DFT
#zero_num = 70 #size of square that's turned to zeroes
#zeroing_mode = 'E' #C: center, E: edge
save_zero_plot = True #if the zeroed plot is saved
save_zero_image = True #if the zeroed image is saved





if color:
    og_image = cv2.imread(image_name)
    image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    if savegray:
        cv2.imwrite(f'gray_{image_name}', image)
else:
    image = cv2.imread(image_name)

imageSmall = cv2.resize(image, (0, 0), fx=0.25, fy=0.25) #Shrinks down image to manageable size


F2_imageSmall = np.fft.fft2(imageSmall.astype(float)) #Does the 2D DFT

if (Dimension == 3):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Y = (np.linspace(-int(imageSmall.shape[0] / 2), int(imageSmall.shape[0] / 2) - 1, imageSmall.shape[0]))
    X = (np.linspace(-int(imageSmall.shape[1] / 2), int(imageSmall.shape[1] / 2) - 1, imageSmall.shape[1]))
    X, Y = np.meshgrid(X, Y)

    if (LogMag):
        ax.plot_surface(X, Y, np.fft.fftshift(np.log(np.abs(F2_imageSmall) + 1)), cmap=plt.cm.coolwarm, linewidth=0,
                        antialiased=False)
    else:
        ax.plot_surface(X, Y, np.fft.fftshift(np.abs(F2_imageSmall)), cmap=plt.cm.coolwarm, linewidth=0,
                        antialiased=False)
    plt.show()

if (Dimension == 2):
    file_inf = ''
    if (LogMag):
        magnitudeImage = np.fft.fftshift(np.log(np.abs(F2_imageSmall) + 1)) #for conversion into display
        magnitudeImage = magnitudeImage / magnitudeImage.max()  # scale to [0, 1]
        magnitudeImage = ski.img_as_ubyte(magnitudeImage)
        file_inf = '_log'
    else:
        magnitudeImage = np.fft.fftshift(np.abs(F2_imageSmall)) #for conversion into display
        magnitudeImage = magnitudeImage / magnitudeImage.max()  # scale to [0, 1]
        magnitudeImage = ski.img_as_ubyte(magnitudeImage)


    if zeroing:
        F2_zeroed_image = F2_imageSmall.copy()
        middleX = int(F2_zeroed_image.shape[0]/2)
        middleY = int(F2_zeroed_image.shape[1]/2)
        zero_file_inf =''
        if (zeroing_mode == 'C'):
            F2_zeroed_image[0:zero_num, 0:zero_num] = 0
            F2_zeroed_image[2*middleX-zero_num:, 0:zero_num] = 0
            F2_zeroed_image[0:zero_num, 2*middleY-zero_num:] = 0
            F2_zeroed_image[2*middleX-zero_num:, 2 * middleY - zero_num:] = 0
            zero_file_inf = f'center_{zero_num}'
        if (zeroing_mode == 'E'):
            F2_zeroed_image[middleX-zero_num:middleX+zero_num, middleY-zero_num:middleY+zero_num] = 0
            zero_file_inf = f'edge_{zero_num}'
        inversed_image = np.fft.ifft2(F2_zeroed_image).astype(np.uint8)
        cv2.imshow("Altered Image", inversed_image)
        cv2.imshow("original Image", imageSmall)

        #displaying the altered DFT
        zero_mag = np.fft.fftshift(np.log(np.abs(F2_zeroed_image)+1))  # for conversion into display
        zero_mag = zero_mag / zero_mag.max()  # scale to [0, 1]
        zero_mag = ski.img_as_ubyte(zero_mag)
        cv2.imshow('zero mag', zero_mag)

        #saving images and plots as files
        if save_zero_plot:
            zero_filename = f'Z_Plot_{zero_file_inf}.jpg'
            cv2.imwrite(zero_filename, zero_mag)
        if save_zero_image:
            zero_filename = f'Z_Image_{zero_file_inf}.jpg'
            cv2.imwrite(zero_filename, inversed_image)


    cv2.imshow('Magnitude plot', magnitudeImage)
    #print(magnitudeImage[:,0])
    cv2.waitKey(0)
    if saveplot:
        filename = f'M_Plot{file_inf}_{image_name}'
        cv2.imwrite(filename, magnitudeImage)
print("Finished!")