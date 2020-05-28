import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

import os
import pickle

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)


def file_info(path):
    f_list = os.listdir(path)
    f_list.sort()

    return f_list    


def check_list(path_dir, path_deslist, path_des, des_list=[], new_list=[], key_list=[], tmp_list=[]):

    file_list = file_info(path_dir)
    for name in file_list:
        rep = name.replace(".jpg","")
        new_list.append(rep)
    new_list.sort()

    file_list = file_info(path_deslist)
    for name in file_list:
        rep = name.replace(".npy","")
        key_list.append(rep)
    key_list.sort()

    file_list = np.load(path_des)[()].keys()
    for name in file_list:
        des_list.append(name)
    des_list.sort()

    if len(new_list) < len(key_list):
        for name in key_list:
            if name not in new_list :
                tmp_list.append(name)
        tmp = 1

    elif len(new_list)>=len(key_list) or len(new_list)>=len(des_list):   
        for name in new_list:
            if name not in key_list or name not in des_list :
                tmp_list.append(name)
        tmp = 0

    return ( tmp, tmp_list )


path_dir = 'img/'
path_des = 'descs/descs.npy'
path_deslist = 'descs/descslist/'
   
( dlt, tmp_list ) = check_list(path_dir, path_deslist, path_des)

img_paths={}
descs={}

if dlt :
    descs = np.load(path_des)[()]
    for name in tmp_list:
        del(descs[name])
        os.remove(path_deslist +name+'.npy')
    np.save(path_des, descs)    

else :
    for name in tmp_list:
        img_paths[name] = path_dir + name + '.jpg'
        descs[name] = 'None'

    for name, img_path in img_paths.items():
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        _, img_shapes, _ = find_faces(img_rgb)
        descs[name] = encode_faces(img_rgb, img_shapes)[0]

        np.save(path_deslist +name+'.npy', descs)

descs.update(np.load(path_des)[()])
np.save(path_des, descs)

print(descs)






