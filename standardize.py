import cv2
import os

i = 1
img_size = 128

path = os.path.join('./assets/dataset','original')

for watch_img in os.listdir(path):
    try:
        img = cv2.imread(os.path.join(path, watch_img))

        h, w = img.shape[:2]
        a1 = w/h
        a2 = h/w

        if(a1 > a2):

            # if width greater than height
            r_img = cv2.resize(img, (round(img_size * a1), img_size), interpolation = cv2.INTER_AREA)
            margin = int(r_img.shape[1]/6)
            crop_img = r_img[0:img_size, margin:(margin+img_size)]

        elif(a1 < a2):

            # if height greater than width
            r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
            margin = int(r_img.shape[0]/6)
            crop_img = r_img[margin:(margin+img_size), 0:img_size]

        elif(a1 == a2):

            # if height and width are equal
            r_img = cv2.resize(img, (img_size, round(img_size * a2)), interpolation = cv2.INTER_AREA)
            crop_img = r_img[0:img_size, 0:img_size]

        if(crop_img.shape[0] != img_size or crop_img.shape[1] != img_size):

            crop_img = r_img[0:img_size, 0:img_size]

        if(crop_img.shape[0] == img_size and crop_img.shape[1] == img_size):

            print("Saving image with dims: " + str(crop_img.shape[0]) + "x" + str(crop_img.shape[1]))
            cv2.imwrite("./assets/dataset/resized/" + str(i + 1) + '.jpg', crop_img)
            i += 1

    except:
        print('Could not save image due to error.')
