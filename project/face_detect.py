import cv2
from collections import Counter
import math
import numpy as np

# relative path is unacceptable
_haar_face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/python/3.6.4_4/Frameworks/Python.framework/'
                                              'Versions/3.6/lib/python3.6/site-packages/cv2/data/'
                                              'haarcascade_frontalface_alt.xml')

def read_pts(file_path):
    with open(file_path, 'r') as f:
        pts = map(lambda x: (float(x[0]), float(x[1])), [c.rstrip().split(' ') for c in f.readlines()[3:-1]])
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
    faces = _haar_face_cascade.detectMultiScale(img, scaleFactor=scaleFactor, minNeighbors=5);
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
        return None,False
    else:
        target_face=sorted(c.items(), key=lambda n: n[1])[-1]
        target_face_id,num=target_face
        return faces[target_face_id],num


def inference_face_from_pts(img_size,pts):
    """
    inferecnce an squre contains all pts
    :param img_size: tuple, (w,h)
    :param pts: iterable, ((x1,y1),(x2,y2)...(xn,yn))
    :return: tuple,(x,y,w,h)
    """

    xs,ys=list(map(lambda x:x[0],pts)),list(map(lambda x: x[1], pts))
    min_x, max_x=math.floor(min(xs)), math.ceil(max(xs))
    min_y, max_y = math.floor(min(ys)), math.ceil(max(ys))
    face=(min_x,min_y,math.ceil(max_x-min_x),math.ceil(max_y-min_y))
    new_face,available=adjust_face(img_size,face,zoom_ratio=1.1)
    return new_face

def adjust_face(img_size,face,zoom_ratio=1.1):
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


