import os
import easyocr
import cv2
import numpy as np
from skimage.util import random_noise

reader = easyocr.Reader(['en'])

def rect_text(img, rect, color, fill=False, text=None):
    x1, y1, x2, y2 = rect

    if color == 'brown':
        outline_color = (42, 42, 165)
        text_color = (255, 255, 255)
        fill_color = (42, 42, 165)
    elif color == 'pink':
        outline_color = (182, 84, 231)
        text_color = (255, 255, 255)
        fill_color = (131, 59, 236)
    elif color == 'yellow':
        outline_color = (55, 250, 250)
        text_color = (255, 0, 0)
        fill_color = (55, 250, 250)
    elif color == 'blue':
        outline_color = (240, 120, 0)
        text_color = (255, 255, 255)
        fill_color = (240, 120, 0)
    elif color == 'green':
        outline_color = (120, 255, 60)
        text_color = (255, 0, 0)
        fill_color = (120, 255, 60)
    elif color == 'orange':
        outline_color = (25, 140, 255)
        text_color = (255, 255, 255)
        fill_color = (25, 140, 255)
    elif color == 'red':
        outline_color = (49, 60, 255)
        text_color = (255, 255, 255)
        fill_color = (49, 60, 255)
    else:
        assert "Color {} not supported".format(color)
        outline_color = (200, 200, 200)
        text_color = (255, 255, 255)
        fill_color = (200, 200, 200)

    if fill is True:
        alpha = 0.7
        overlay = img.copy()
        overlay[y1:y2, x1:x2] = fill_color
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    if text is not None:
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1.6
        font_thickness = 4
        box_size, _ = cv2.getTextSize(text, font_type, font_size, font_thickness)
        txt_size = box_size[0]
        txt_loc = (int(x1 + (x2 - x1)/2 - txt_size/2), y2 - 15)
        cv2.putText(img, text, txt_loc, font_type, font_size, font_thickness)


def process_ocr(file_name, rects):

    file_name_base = os.path.basename(file_name).split('.')[0]
    img = cv2.imread(file_name)
    img2 = img.copy()

    for idx, rect in enumerate(rects):

        # Finding bounding box
        x1, y1, x2, y2 = rect

        # Cropped img input for analysis
        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(f'{file_name_base}_{idx}_cropped.jpg', cropped)

        # Grayscale the cropped
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f'{file_name_base}_{idx}_gray.jpg', gray)

        # Guassian blur
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        cv2.imwrite(f'{file_name_base}_{idx}_blur.jpg', blur)

        # Mediaun blur
        noise_img = random_noise(blur, mode="s&p", amount=0.2)
        noise_img = np.array(255*noise_img, dtype = 'uint8')
        median = cv2.medianBlur(noise_img, 5)
        cv2.imwrite(f'{file_name_base}_{idx}_median_blur.jpg', median)

        # Binarize the image
        _, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        cv2.imwrite(f'{file_name_base}_{idx}_otsu.jpg', otsu)
        # text1 = reader.readtext(otsu, detail=0)
        # print(text1)
        # text1 = ' '.join(text1)

        # Dilate the image
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
        kernel = np.ones((3, 3))
        dilation = cv2.dilate(otsu, kernel, iterations=1)
        cv2.imwrite(f'{file_name_base}_{idx}_dilation2.jpg', dilation)
        text = reader.readtext(dilation, detail=0)
        print(text)

        # Drawing a rectangle on copied image
        rect_text(img2, [x1, y2+10, x2, y2+70], color='yellow', fill=True, text=text)
        # rect_text(img2, [x1, y2+75, x2, y2+135], color='yellow', fill=True, text=text2.upper())

    # Save the image
    processed_fname = f'{file_name_base}_processed.jpg'
    cv2.imwrite(processed_fname, img2)


if __name__ == '__main__':
    img_rects = {
        '1.jpg': [
                [75, 365, 1520, 444],
                [80, 680, 1535, 765]
            ],
        # '2.jpg': [
        #         [104, 396, 1485, 488],
        #         [90, 745, 1491, 831]
        #     ],
        # '3.jpg': [
        #         [120, 360, 1337, 431],
        #         [160, 592, 1391, 698]
        #     ],
        '4.jpg': [
                [65, 307, 1493, 390],
                [59, 618, 1517, 693]
            ],
    }

    for img_fname in img_rects.keys():
        process_ocr(img_fname, img_rects[img_fname])
