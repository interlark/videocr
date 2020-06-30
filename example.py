import subprocess
import warnings
from tempfile import NamedTemporaryFile

import numpy as np
from videocr import get_subtitles
from wand.image import Image
import os
import cv2 

SUBTITLES_COLOR = 'ffffff'


def crop_bbox(img_path):
    img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold
    thresh = cv2.adaptiveThreshold(gray_image, 255, 1, 1, 11, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    thresh = cv2.dilate(thresh,None,iterations = 15)
    thresh = cv2.erode(thresh,None,iterations = 15)

    # Find contours
    contours, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_countur = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(max_countur)

        cropped_img = img[y:y+h, x:x+w]
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # cv2.imshow('Contours', cropped_img)
        # cv2.waitKey(0)

        # cv2.imwrite('test.png', cropped_img)
        cv2.imwrite(img_path, cropped_img)


def img_hook(img):
    with Image.from_array(img) as w_img, NamedTemporaryFile(suffix='.png') as tmp:
        w_img.save(filename=tmp.name)
        subprocess.call(f'mogrify -fill black -fuzz 15% +opaque "#{SUBTITLES_COLOR}" ' + tmp.name, shell=True)
        crop_bbox(tmp.name)

        with Image(filename=tmp.name) as w_img_out, warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = np.array(w_img_out)
            # with Image.from_array(result) as w_test:
            #     w_test.save(filename=f'check_hook/{os.path.basename(tmp.name)}')
            return result

if __name__ == '__main__':
    # use_fullframe = True, because we're already cropped bottom part of a screencast.
    subs_ocred = get_subtitles('example.mp4', lang='rus', sim_threshold=90, conf_threshold=65, 
        use_fullframe=True, download_best=True, img_hook=img_hook, take_every_nth_frame=1, processes=4)
    with open('ocred_subtitles.srt', 'w') as f_srt:
        f_srt.write(subs_ocred)
