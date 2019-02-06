import cv2
import numpy as np
import skimage as ski


print("Enter the name of the file:")
image_name = input()

print("Will it be a low-pass or high-pass filter?")
filter_dict = {"high": True, "highpass": True, "h": True, "high-pass":True, "low": False, "lowpass": False, "l": False,
               "low-pass": False}

invalid = True
while (invalid):
    usr_input = input()
    usr_input = usr_input.lower()
    if usr_input in filter_dict:
        invalid = False
highpass = filter_dict[usr_input]


#settings
color = True #If the image is in color, it would need to be converted
savegray = True #Will the gray image be shared
#highpass = True #True if high pass filter, false if low pass
save_ideal = True #save the ideal pass image and filter as .jpgs
save_butterworth = True #save the butterworth filters and images as .jpgs

#image_name = 'DSC_9259-0.50.JPG'


print("Running now...")
if color:
    og_image = cv2.imread(image_name)
    image = cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY)
    if savegray:
        cv2.imwrite(f'gray_{image_name}', image)
else:
    image = cv2.imread(image_name)

imageSmall = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)  # Shrinks down image to manageable size

pass_mode = 'low'
if highpass:
    pass_mode = 'high'

Y = (np.linspace(-int(imageSmall.shape[0]/2), int(imageSmall.shape[0]/2)-1, imageSmall.shape[0]))
X = (np.linspace(-int(imageSmall.shape[1]/2), int(imageSmall.shape[1]/2)-1, imageSmall.shape[1]))
X, Y = np.meshgrid(X, Y) #creates a grid for use in creating filter

D = np.sqrt(X*X + Y*Y) #all of the diagonals
D0 = 0.25 * D.max()  #function of the greatest diagonal, so the filter is going to be a function of a quarter of that
if highpass:
    idealPass = D >= D0
else:
    idealPass = D <= D0

FTimageSmall = np.fft.fft2(imageSmall.astype(float))
FTimageSmallFiltered = FTimageSmall * np.fft.fftshift(idealPass)
imageSmallFiltered = np.abs(np.fft.ifft2(FTimageSmallFiltered))

idealPass = ski.img_as_ubyte(idealPass / idealPass.max()) #Creates the image of the filter
imageSmallFiltered = ski.img_as_ubyte(imageSmallFiltered / imageSmallFiltered.max()) #creates the image post filter
cv2.imshow("Ideal Passed Image", imageSmallFiltered)
cv2.imshow("Ideal Pass", idealPass)
print("Ideal Images Generated")
cv2.waitKey(0)
cv2.destroyAllWindows()
if save_ideal: #optional: saves the images
    cv2.imwrite('ideal_'+pass_mode+'_pass.jpg', idealPass)
    cv2.imwrite('ideal_'+pass_mode+'_passed_image.jpg', imageSmallFiltered)


for n in range(1, 5):
    # Create Butterworth filter of order n
    if highpass:
        H = (1 + (np.sqrt(2) - 1) * np.power(D / D0, n))
    else:
        H = 1.0 / (1 + (np.sqrt(2) - 1) * np.power(D / D0, 2 * n))
    # Apply the filter to the grayscaled image
    FTimageSmallFiltered = FTimageSmall * np.fft.fftshift(H)
    imageSmallFiltered = np.abs(np.fft.ifft2(FTimageSmallFiltered))
    imageSmallFiltered = ski.img_as_ubyte(imageSmallFiltered / imageSmallFiltered.max())
    if save_butterworth:
        cv2.imwrite("image_butterworth_"+ pass_mode +'_' + str(n) + ".jpg", imageSmallFiltered)
    cv2.imshow("image_butterworth_"+ pass_mode +'_' + str(n), imageSmallFiltered);

    H = ski.img_as_ubyte(H / H.max())
    cv2.imshow('H', H)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if save_butterworth:
        cv2.imwrite("butterworth_" + pass_mode + '_' + str(n) + ".jpg", H)

cv2.waitKey(0)
cv2.destroyAllWindows()