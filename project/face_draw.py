import cv2
from project import face_detect
from project import *


def draw_faces(img, faces, color=MARK_COLOR_GREEN):
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

def draw_landmarks(img, pts, radius,color=MARK_COLOR_GREEN):
    img_w,img_h=img.shape[:2]

    for x, y in pts:

        if 0<=x<=1 and 0<=y<=1:
            x*=img_w
            y*=img_h

        x,y=int(round(x)),int(round(y))

        cv2.circle(img, (x,y), radius, color, -1)


def display_landmark(targets):
    MAX_HEIGHT=960
    MAX_WEIGHT=1260

    max_zoom_ratio=1.0

    """
    display targets in an grid
    :param targets: Iterable (url:pts).
    :return: None
    """
    for url,pts in targets:
        img=cv2.imread(url)
        height, weight = img.shape[:2]

        big_height,big_wight=height > MAX_HEIGHT,weight > MAX_WEIGHT
        if big_height or big_wight:
            radius=3
            zoom_type='!'
        else:
            radius=2
            zoom_type='.'

        if big_wight or big_height:
            ratio= min(MAX_HEIGHT/height,MAX_WEIGHT/weight)
            height=round(height*ratio)
            weight=round(weight*ratio)

        print('%s [%s] ' % (zoom_type, url))

        draw_landmarks(img, pts, radius)

        # filter and draw face
        faces = face_detect.detect_face(img)
        face,pts_num_contained= face_detect.filter_face(faces, pts)

        if pts_num_contained< len(pts)/2:
            # if the detected face does not include most pts, it is invalid
            face=None
        else:
            contains_all=(len(pts)==pts_num_contained)

        img_size = img.shape[:2]
        if face is None:

            face= face_detect.inference_face_from_pts(img_size, pts)
            # inference face from pts
            draw_faces(img, [face], MARK_COLOR_WHITE)
        else:

            # original face
            draw_faces(img, [face], MARK_COLOR_PURPLE)

            zoom_ratio=1.0
            while not contains_all:
                face,can_adjust = face_detect.adjust_face(img_size, face, zoom_ratio)
                if not can_adjust:
                    break
                _, pts_num_contained = face_detect.filter_face([face], pts)
                if len(pts)==pts_num_contained:
                    break
                zoom_ratio += 0.1

            # print('zoom ratio=%.3f'%zoom_ratio)
            if zoom_ratio>max_zoom_ratio :
                max_zoom_ratio = zoom_ratio

            # adjusted face
            if not can_adjust:
                draw_faces(img, [face], MARK_COLOR_RED)
            else:
                draw_faces(img, [face], MARK_COLOR_GREEN)

        # resize img
        img = cv2.resize(img, (weight, height)) if zoom_type == '!' else img

        # create a named window and move it
        cv2.namedWindow(url)
        cv2.moveWindow(url,0,0)

        cv2.imshow(url,img)
        cv2.waitKey(0)
        cv2.destroyWindow(url)

    print ('max_zoom_ratio=%.3f'%max_zoom_ratio)

if __name__ == '__main__':
    import os
    import random

    def _read_targets(root='../data/lfpw/testset',file=None,num=9):
        target=[]

        for root, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.pts'):
                    f = filename.split('.')
                    if file is None or file in f[0]:
                        jpg=os.path.join(root,f[0]+'.jpg')
                        png=os.path.join(root,f[0]+'.png')
                        if os.path.isfile(jpg):
                            url=jpg
                        elif os.path.isfile(png):
                            url=png
                        else:
                            raise ValueError('unknown file: %s'%os.path.join(root,filename))

                        target.append((url,list(face_detect.read_pts(os.path.join(root, filename)))))
        if not num is None:
            random.shuffle(target)
            target=target[:num]
        return target

    # targets=_read_targets(root='../data/300VW_Dataset_2015_12_14/007',num=20)
    targets = _read_targets(root='../data/300VW/007', num=20)
    # targets = _read_targets(root='../data/afw/',file='17191', num=20)
    # targets = _read_targets(root='../data/afw/', file='17191_17', num=20)
    # targets = _read_targets(root='../data/afw/', file='17191_2', num=20)
    display_landmark(targets)
    print('%d image displayed' % len(targets))

