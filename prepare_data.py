import face_detect as detect
import os
import cv2
import random
import tensorflow as tf
from collections import namedtuple


IMG_SIZE=224
ZOOM_RATIO=1.5

def print_pts(pts):
    msg=['version: 1','n_points:  %d'%len(pts),'{']
    for pt in pts:
        msg.append('%.6f %.6f'%(pt[0],pt[1]))
    msg.append('}')
    return '\n'.join(msg)

def resize_img(img,size):
    return cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)


def crop_img(img,face):
    x, y, w, h = face
    return img[y:y+h,x:x+w]


def crop(img,face,pts,size):
    """
    crop face from img, resize it and transfer pts accordingly
    :param img:
    :param face:
    :param pts:
    :param size:
    :return: (new_img,new_pts)
    """
    new_img =crop_img(img,face)
    new_img=resize_img(new_img,size)
    x, y, w, h = face
    new_pts= list(map(lambda pt:((pt[0]-x)/w, (pt[1]-y)/h),pts))

    return new_img,new_pts

def decode_from_tfrecords(filename_queue, batch_size, shuffle):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'source_filename': tf.FixedLenFeature([], tf.string),
                                           'crop_filename': tf.FixedLenFeature([], tf.string),
                                           'image/encoded': tf.FixedLenFeature([], tf.string),
                                           'image/source_face': tf.FixedLenFeature([4], tf.int64),
                                           'label/points': tf.FixedLenFeature([136], tf.float32)
                                       })
    image_raw = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.reshape(image_raw, [IMG_SIZE, IMG_SIZE, 3])
    pts = features['label/points']
    source_filename = features['source_filename']
    crop_filename = features['crop_filename']
    source_face = features['image/source_face']

    assert batch_size >0
    capacity = 1 * batch_size
    min_after_dequeue = 1 * batch_size

    if shuffle:
        image_value, pts_value, source_filename_value,crop_filename_value,source_face_value = tf.train.shuffle_batch([image, pts,source_filename,crop_filename,source_face],
                                          batch_size=batch_size,
                                          num_threads=8,
                                          capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)
    else:
        image_value, pts_value, source_filename_value, crop_filename_value, source_face_value = tf.train.batch([image, pts,source_filename,crop_filename,source_face],
                             batch_size=batch_size,
                             num_threads=8,
                             capacity=capacity)

    return image_value, pts_value, source_filename_value,crop_filename_value,source_face_value


def crop_all(from_root_path, from_dirs, match_names, output_dir):

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        raise RuntimeError('can not find dir %s' % output_dir)

    for from_dir in from_dirs:
        for root, _, filenames in os.walk(os.path.join(from_root_path, from_dir)):

            count = 1
            invalid_pts_count = 0
            inferenced_from_pts_count=0
            total_count = len(filenames) / 2

            for filename in filenames:

                if filename.endswith('.pts'):
                    f = filename.split('.')
                    if not match_names is None and len(match_names)!=0 and not f[0] in match_names:
                        continue

                    jpg = os.path.join(root, f[0] + '.jpg')
                    png = os.path.join(root, f[0] + '.png')
                    if os.path.isfile(jpg):
                        url = jpg
                    elif os.path.isfile(png):
                        url = png
                    else:
                        raise ValueError('unknown file: %s' % os.path.join(root, filename))

                    path = os.path.join(root, filename)
                    pts = list(detect.read_pts(path))

                    def should_reserve(x):
                        return x != 'data' and x != '..' and x != 'image'

                    if count % 10 == 0:
                        print('[%s] %d/%d' % (root, count, total_count))

                    count += 1

                    if [1 for pt in pts if pt[0] < 0 or pt[1] < 0]:
                        # if exists invalid(negative) pt, skip
                        invalid_pts_count += 1
                        continue

                    new_f = '_'.join(filter(should_reserve, os.path.join(root, f[0]).split(os.sep)))
                    meta_path, img_path, pts_path, = os.path.join(output_dir, new_f + '.meta'), os.path.join(output_dir,
                                                                                                             new_f + '.jpg'), os.path.join(
                        output_dir, new_f + '.pts')

                    if not os.path.exists(img_path):
                        # read img file
                        img = cv2.imread(url)

                        # filter face
                        faces = detect.detect_face(img)
                        if len(faces)==0:
                            # can not detect face, use face inerenced from pts
                            inferenced_from_pts_count+=1
                            face=detect.inference_face_from_pts(img_size=img.shape[:2],pts=pts)

                        else:
                            face, pts_num_contained = detect.filter_face(faces, pts)
                            if pts_num_contained < len(pts) / 2:
                                # if the detected face does not include most pts, it is invalid, use face inerenced from pts
                                inferenced_from_pts_count += 1
                                face = detect.inference_face_from_pts(img_size=img.shape[:2], pts=pts)

                        zoom_ratio=1.0
                        while True:
                            face, can_adjust = detect.adjust_face(img.shape[0:2], face, zoom_ratio)

                            if not can_adjust:
                                # adjusted face is out of boundary. use face inerenced from pts
                                inferenced_from_pts_count += 1
                                face = detect.inference_face_from_pts(img_size=img.shape[:2], pts=pts)
                                break

                            _, pts_num_contained = detect.filter_face([face], pts)
                            if len(pts) != pts_num_contained:
                                # some pts is still out of current face, enlarge face
                                zoom_ratio += 0.1
                            else:
                                #current face contains all pts, use it
                                break

                        new_img, new_pts = crop(img, face, pts, size=IMG_SIZE)

                        # write pts file and meta file first
                        with open(pts_path, 'w') as pts_file:
                            pts_file.write(print_pts(new_pts))

                        with open(meta_path, 'w') as meta_file:
                            src_img_path = url
                            if src_img_path.startswith('../'):
                                src_img_path = src_img_path[3:]

                            # meta file contains two lines
                            # line 1 : raw image path
                            # line 2 : face
                            meta_file.write("%s\n%d,%d,%d,%d" % (src_img_path, *face))

                        # then write img, thus if img_path does not exists, it is guaranteed to generate pts file
                        cv2.imwrite(img_path, new_img)                  
    return invalid_pts_count,inferenced_from_pts_count,total_count


SplitEntry=namedtuple('SplitEntry',['target_path','ratio'])

SplitSpec=namedtuple('SplitSpec',['from_dir','shuffle','splitEntries'])


def write_tf_file(tf_file_name,url_files, index):
    total_length = len(index)

    with tf.python_io.TFRecordWriter(tf_file_name) as writer:
        for idx, value in enumerate(index):
            url= url_files[value]

            file = os.path.splitext(url)[0]

            meta = file + ".meta"
            pts_file = file +".pts"
            pts = detect.read_pts_as_float_list(pts_file)
            img = cv2.imread(url)
            img_raw = img.tostring()

            with open(meta, 'r') as meta_file:
                meta_content = meta_file.read()
                src_img_path, face = meta_content.split('\n')
                src_img_path =bytes(src_img_path, encoding='ascii')
                face = list(map(int,face.split(',')))
                crop_filename = bytes(url.split('/')[-1], encoding='ascii')

            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'source_filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[src_img_path])),
                    'crop_filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[crop_filename])),
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                    'image/source_face': tf.train.Feature(int64_list=tf.train.Int64List(value=face)),
                    'label/points': tf.train.Feature(float_list=tf.train.FloatList(value=pts)),
                }))
            writer.write(example.SerializeToString())
            if idx>=500 and idx % 500 == 0:
                print('[{name} {now}/{total}]'.format(name=tf_file_name.split('/')[-1], now=idx, total=total_length))
    # end
    print('[{name} {total}/{total}]'.format(name=tf_file_name.split('/')[-1], total=total_length))


def compact_all(split_spec):
    f_list=os.listdir(split_spec.from_dir)
    url_files = []
    for filename in f_list:
        sp=os.path.splitext(filename)
        if sp[1]=='.jpg' or sp[1]=='.png':
            url_files.append(os.path.join(split_spec.from_dir, filename))

    count=len(url_files)
    idx=list(range(count))
    if split_spec.shuffle :
        random.shuffle(idx)

    count_sum=0

    # verify first

    ratio_sum=0
    for splitEntry in split_spec.splitEntries:
        ratio = splitEntry.ratio
        if ratio <=0:
            raise ValueError('invalid ratio %s'%ratio)
        ratio_sum += ratio

    if ratio_sum !=1:
        raise ValueError('invalid ratio_sum %s' % ratio_sum)

    for id,splitEntry in enumerate(split_spec.splitEntries):
        target_path,ratio= splitEntry.target_path,splitEntry.ratio

        if id == len(split_spec.splitEntries)-1:
            # last
            index=idx[count_sum:]
        else:
            c=round(ratio*count)
            index=idx[count_sum:count_sum+c]
            count_sum+=c

        write_tf_file(target_path, url_files,index)


if __name__=='__main__':

    # from collections import Counter
    # c = Counter()
    # for dir in ['300VW', '300W', 'afw', 'helen', 'ibug', 'lfpw']:
    #     for _, _, filenames in os.walk(os.path.join('data', dir)):
    #         for filename in filenames:
    #             c.update([filename.split('.')[-1]])
    # print('data postfix : %s' % c)

    def verify_tfrecords(file,shuffle=False):
        filename_queue = tf.train.string_input_producer([file])
        image_value, pts_value, source_filename_value, crop_filename_value, source_face_value =decode_from_tfrecords(filename_queue,batch_size=1,shuffle=shuffle)

        with tf.Session() as sess:
            init_op=tf.initialize_all_variables()
            sess.run(init_op)
            coord=tf.train.Coordinator() #创建一个协调器，管理线程
            threads=tf.train.start_queue_runners(coord=coord) #启动QueueRunner, 此时文件名队列已经进队。

            for i in range(4):
                iv,pv,sfv,cfv,sfv2=(sess.run([image_value, pts_value, source_filename_value,crop_filename_value,source_face_value]))
                print('[source_file=%s,crop_file=%s, source_face=%s, pts.shape=%s img.shape=%s'%(sfv,cfv,sfv2,pv.shape,iv.shape))

            coord.request_stop()
            coord.join(threads)


    invalid_count,inferenced_from_pts_count,total_count=crop_all(from_root_path='data'
         , from_dirs=['300VW', '300W', 'afw', 'helen', 'ibug', 'lfpw']
         # , from_dirs = ['ibug']
         , match_names =[]
         , output_dir='data/output')

    print('total_count=%d, invalid_count=%d, inferenced_from_pts_count=%d' % (total_count,invalid_count,inferenced_from_pts_count))

    split_spec=SplitSpec(from_dir = 'data/output',shuffle=True,splitEntries=[SplitEntry(target_path='data/tfrecords/train',
                                                                               ratio=0.95),
                                                                    SplitEntry(target_path='data/tfrecords/validate',
                                                                               ratio=0.05)])

    compact_all(split_spec)

    verify_tfrecords(file='data/tfrecords/train')


