import cv2
from collections import Counter
import math
import numpy as np
import face_recognition

# relative path is unacceptable
_haar_face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/python/3.6.4_4/Frameworks/Python.framework/'
                                              'Versions/3.6/lib/python3.6/site-packages/cv2/data/'
                                              'haarcascade_frontalface_alt.xml')

def read_pts(file_path):
    with open(file_path, 'r') as f:
        pts = list(map(lambda x: (float(x[0]), float(x[1])), [c.rstrip().split(' ') for c in f.readlines()[3:-1]]))
        return pts


def read_pts_as_float_list(file_path):
    with open(file_path, 'r') as f:
        pts=[]
        for x  in  [c.rstrip().split(' ') for c in f.readlines()[3:-1]]:
            pts.append(float(x[0]))
            pts.append(float(x[1]))
        return pts


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def detect_face(img,scaleFactor=1.1):
    faces = face_recognition.face_locations(img)
    # conv (top, right, bottom, left) to (x,y,w,h),eg: (left,top,right-left,bottom-top)
    faces = [(face[3],face[0],face[1]-face[3],face[2]-face[0]) for face in faces]
    # faces = _haar_face_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=5);
    return faces

def pt_in_face(pt,face):
    x,y,w,h=face
    px,py=pt
    return x<px<x+w and y<py<y+h

def has_invalid_pt(img_size,pts):
    img_h, img_w = img_size

    for x, y in pts:
        if x<0 or x> img_w or y<0 or y>img_h:
            return False

    return True

def filter_face(faces,pts):
    """
    filter the face fit most pts
    :param faces:
    :param pts:
    :return: (the face with most pts, the pts it contains)
    """
    c = Counter()
    c.update([id for id, face in enumerate(faces) for pt in pts if pt_in_face(pt,face)])
    if not c:
        return None,0
    else:
        target_face=sorted(c.items(), key=lambda n: n[1])[-1]
        target_face_id,num=target_face
        return faces[target_face_id],num


def inference_face_from_pts(img_size,pts,zoom_ratio=1.05):
    """
    inferecnce an square contains all pts
    :param img_size: tuple, (w,h)
    :param pts: iterable, ((x1,y1),(x2,y2)...(xn,yn))
    :param zoom_ratio: zoom_ratio
    :return: tuple,(x,y,w,h)
    """

    xs,ys=list(map(lambda x:x[0],pts)),list(map(lambda x: x[1], pts))
    min_x, max_x=math.floor(min(xs)), math.ceil(max(xs))
    min_y, max_y = math.floor(min(ys)), math.ceil(max(ys))
    face=(min_x,min_y,math.ceil(max_x-min_x),math.ceil(max_y-min_y))
    new_face,available=adjust_face(img_size,face,zoom_ratio)
    return new_face

def adjust_face(img_size,face,zoom_ratio):
    """
    return a squre face contains (rectangle) face
    :param img_size: tuple, (w,h)
    :param face: (rectangle) face, (x,y,w,h)
    :param zoom_ratio: zoom new face
    :return: (face,available), face:(x,y,w,h), available:boolean
    """

    def resize_axis(start,length,new_length,total):
        # avoid overflow
        if new_length>total:
            return None,False

        if length < new_length:
            delta_1 = round((new_length - length) / 2)
            delta_2 = new_length - length - delta_1
            start -= delta_1
            if start < 0:
                delta_2 += abs(start)
                start = 0

            length += delta_2

            if length > total:
                delta = length - total
                if start > delta:
                    start -= delta
                else:
                    return None,False

        return np.int32(round(start)),True

    img_h,img_w=img_size
    x,y,w,h = face
    size = math.ceil(max(w, h) * zoom_ratio)
    new_x,x_available= resize_axis(x,w,size,img_w)
    new_y,y_available= resize_axis(y,h, size, img_h)

    if x_available and y_available:
        return (new_x, new_y, size, size),True
    else:
        return (x,y,w,h),False


if __name__ == '__main__':


    # img_name='data/ibug/image_010_1.jpg'
    # pts_name= img_name.split('.')[0]+'.pts'
    # img=face_recognition.load_image_file(img_name)
    # faces_locations=face_recognition.face_locations(img)
    # print(faces_locations)

    import cv2
    import os
    img_name='data/ibug/image_050_01.jpg'
    pts_name= img_name.split('.')[0]+'.pts'

    img = cv2.imread(img_name)

    faces =detect_face(img)


    if (os.path.exists(pts_name)):
        pts =read_pts(pts_name)
        marks = np.reshape(pts, (-1, 2))
        for mark in marks:
            cv2.circle(img, (int(mark[0]), int(
                mark[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)

    for face in faces:
        # face, can_adjust = adjust_face(img.shape[0:2], face, 1.5)

        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)

    cv2.imshow('result', img)
    cv2.waitKey()


