from project import face_detect as detect
import os
import cv2
import random
import tensorflow as tf


def print_pts(pts):
    msg=['version: 1','n_points:  %d'%len(pts),'{']
    for pt in pts:
        msg.append('%.6f %.6f'%(pt[0],pt[1]))
    msg.append('}')
    return '\n'.join(msg)

def crop(img,face,pts,size=128):
    """
    crop face from img, resize it and transfer pts accordingly
    :param img:
    :param face:
    :param pts:
    :param size:
    :return: (new_img,new_pts)
    """
    x, y, w, h = face
    new_img=img[y:y+h,x:x+w]
    new_img=cv2.resize(new_img,(size,size),interpolation=cv2.INTER_CUBIC)
    new_pts= list(map(lambda pt:((pt[0]-x)/w, (pt[1]-y)/h),pts))

    return new_img,new_pts

def decode_from_tfrecords(filename_queue, batch_size, shuffle):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'pts': tf.FixedLenFeature([136], tf.float32),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                           'file': tf.FixedLenFeature([], tf.string),
                                       })
    image_raw = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image_raw, [128, 128, 3])
    pts = features['pts']
    file = features['file']


    assert batch_size >0
    capacity =  20 * batch_size
    min_after_dequeue =10*batch_size




    if shuffle:
        i, p, f = tf.train.shuffle_batch([image, pts,file],
                                          batch_size=batch_size,
                                          num_threads=8,
                                          capacity=capacity,
                                          min_after_dequeue=min_after_dequeue)
    else:
        i, p, f = tf.train.batch([image, pts, file],
                             batch_size=batch_size,
                             num_threads=8,
                             capacity=capacity)

    return i,p,f

def find_path_from_url(url):
    url=url.split('_')
    root='data'

    if url[0]=='300W':
        path=os.sep.join([root, url[0], '_'.join(url[2:])])
    elif url[0]=='300VW':
        path = os.sep.join([root,url[0],url[1],'image',url[2]])
    elif url[0]=='afw':
        path = os.sep.join([root, url[0], '_'.join(url[1:])])
    elif url[0]=='helen':
        path = os.sep.join([root, url[0],url[1], '_'.join(url[2:])])
    elif url[0]=='ibug':
        path = os.sep.join([root, url[0],'_'.join(url[1:])])
    elif url[0]=='lfpw':
        path = os.sep.join([root, url[0],url[1],'_'.join(url[2:])])

    return path

if __name__=='__main__':

    def crop_all():
        from_root_path = '../data'
        from_dirs = ['300VW', '300W', 'afw', 'helen', 'ibug', 'lfpw']
        target_dir = '../data/output'
        max_zoom_ratio = 1.0

        if not os.path.isdir(target_dir):
            raise RuntimeError('can not find dir %s'%target_dir)


        for from_dir in from_dirs:
            for root, _, filenames in os.walk(os.path.join(from_root_path,from_dir)):

                count = 1
                invalid_count=0
                total_count=len(filenames)/2

                for filename in filenames:

                    if filename.endswith('.pts'):
                        f = filename.split('.')
                        jpg=os.path.join(root,f[0]+'.jpg')
                        png=os.path.join(root,f[0]+'.png')
                        if os.path.isfile(jpg):
                            url=jpg
                        elif os.path.isfile(png):
                            url=png
                        else:
                            raise ValueError('unknown file: %s'%os.path.join(root,filename))

                        pts=list(detect.read_pts(os.path.join(root, filename)))

                        def should_reserve(x):
                            return x!='data' and x!='..' and x!='image'

                        if count % 100 == 0:
                            print('[%s] %d/%d' % (root, count, total_count))

                        count += 1

                        if [1 for pt in pts if pt[0] < 0 or pt[1]<0]:
                            # if exists invalid(negative) pt, skip
                            invalid_count+=1
                            continue

                        new_f='_'.join(filter(should_reserve,os.path.join(root,f[0]).split(os.sep)))

                        img_path, pts_path = os.path.join(target_dir, new_f + '.jpg'),os.path.join(target_dir, new_f + '.pts')

                        if os.path.exists(img_path) and os.path.exists(pts_path):
                            continue

                        img = cv2.imread(url)

                        # filter face
                        faces = detect.detect_face(img)
                        face, pts_num_contained = detect.filter_face(faces, pts)

                        if pts_num_contained < len(pts) / 2:
                            # if the detected face does not include most pts, it is invalid
                            face = None
                        else:
                            contains_all = (len(pts) == pts_num_contained)

                        img_size = img.shape[:2]
                        if face is None:
                            face = detect.inference_face_from_pts(img_size, pts)
                        else:
                            zoom_ratio = 1.1
                            while not contains_all:
                                face, can_adjust = detect.adjust_face(img_size, face, zoom_ratio)
                                if not can_adjust:
                                    face = detect.inference_face_from_pts(img_size, pts)
                                    break
                                _, pts_num_contained = detect.filter_face([face], pts)
                                if len(pts) == pts_num_contained:
                                    break
                                zoom_ratio += 0.1

                            # print('zoom ratio=%.3f'%zoom_ratio)
                            if zoom_ratio > max_zoom_ratio:
                                print('inc max zoom_ration to %.3f'%zoom_ratio)
                                max_zoom_ratio = zoom_ratio

                        new_img,new_pts=crop(img,face,pts,size=128)

                        cv2.imwrite(img_path, new_img)
                        with open(pts_path, 'w') as pts_file:
                            pts_file.write(print_pts(new_pts))

                        # draw.draw_landmarks(new_img,new_pts,1)
                        # cv2.imshow(url, new_img)
                        # cv2.waitKey(0)
                        # cv2.destroyWindow(url)

        print('max zoom_ratio is %.3f, invalid_count=%d'%(max_zoom_ratio,invalid_count))

    def compact_all():
        from_dir = '../data/output'

        train_path='../data/train.tfrecords'
        test_path = '../data/test.tfrecords'
        validate_path = '../data/validate.tfrecords'
        f_list=os.listdir(from_dir)
        pts_files = []
        for filename in f_list:
            sp=os.path.splitext(filename)
            if sp[1]=='.pts':
                pts_files.append(os.path.join(from_dir, filename))

        count=len(pts_files)
        idx=list(range(count))
        random.shuffle(idx)

        train_count=round(count*0.8)
        test_count=round(count*0.1)

        def write_tf_file(tf_file_name,index):
            total_length=len(index)

            with tf.python_io.TFRecordWriter(tf_file_name) as writer:
                for idx, value in enumerate(index):
                    pts_file=pts_files[value]
                    file=os.path.splitext(pts_file)[0]
                    jpg = file + '.jpg'
                    png = file + '.png'

                    if os.path.isfile(jpg):
                        url = jpg
                    elif os.path.isfile(png):
                        url = png
                    else:
                        raise ValueError('unknown file: %s' % os.path.join(from_dir, filename))

                    pts = detect.read_pts_as_float_list(pts_file)
                    img = cv2.imread(url)
                    img_raw=img.tostring()

                    file_raw=bytes(url.split('/')[-1],encoding='ascii')

                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'pts': tf.train.Feature(float_list=tf.train.FloatList(value=pts)),
                            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            'file':tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_raw]))
                        }))
                    writer.write(example.SerializeToString())
                    if idx % 500 == 1:
                        print('[{name} {now}/{total}]'.format(name=tf_file_name.split('/')[-1],now=idx,total=total_length))

        # write_tf_file('../data/demo.tfrecords', [1,2,3])
        write_tf_file(train_path, idx[:train_count])
        write_tf_file(test_path, idx[train_count:train_count+test_count])
        write_tf_file(validate_path, idx[train_count + test_count:])


    def verify_tfrecords(file,shuffle):
        filename_queue = tf.train.string_input_producer([file])
        image,pts,file=decode_from_tfrecords(filename_queue,batch_size=2,shuffle=shuffle)
        print('file %s'%file)

        with tf.Session() as sess:
            init_op=tf.initialize_all_variables()
            sess.run(init_op)
            coord=tf.train.Coordinator() #创建一个协调器，管理线程
            threads=tf.train.start_queue_runners(coord=coord) #启动QueueRunner, 此时文件名队列已经进队。


            for i in range(2):
                m,p,f=(sess.run([image,pts,file]))
                print('[file=%s p.shape=%s m.shape=%s'%(f,p.shape,m.shape))

            coord.request_stop()
            coord.join(threads)


    # crop_all()
    # compact_all()
    verify_tfrecords('../data/demo.tfrecords')