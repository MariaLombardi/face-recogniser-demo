#!/usr/bin/python3

import numpy as np
import json
import os
import cv2

from utilities import extract_faces


def read_openpose_from_json(json_filename):

    with open(json_filename) as data_file:
        loaded = json.load(data_file)

        poses = []
        conf_poses = []
        faces = []
        conf_faces = []

        for arr in loaded["people"]:
            conf_poses.append(arr["pose_keypoints_2d"][2::3])
            arr_poses = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
            poses.append(list(zip(arr_poses[::2], arr_poses[1::2])))

            conf_faces.append(arr["face_keypoints_2d"][2::3])
            arr_faces = np.delete(arr["face_keypoints_2d"], slice(2, None, 3))  # remove confidence values from the array
            faces.append(list(zip(arr_faces[::2], arr_faces[1::2])))

    return poses, conf_poses, faces, conf_faces


root_dir = os.path.join(os.getcwd(), '../../dataset')
lfw_dataset_dir = os.path.join(root_dir, 'LFW')
cut_lfw_dataset_dir = os.path.join(root_dir, 'LFW_cut')
lfw_json_dir = os.path.join(root_dir, 'LFW_json')

samples = [sample.replace('_keypoints.json', '') for sample in os.listdir(lfw_json_dir)]
samples.sort()
for sample in samples:
    openpose_file = os.path.join(lfw_json_dir, sample + '_keypoints.json')
    poses, conf_poses, faces, conf_faces = read_openpose_from_json(openpose_file)
    if len(poses) == 1:
        img_file = os.path.join(lfw_dataset_dir, sample + '.jpg')
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        faces_img, bboxes, order = extract_faces(img, poses, required_size=(160, 160))
        cv2.imwrite(cut_lfw_dataset_dir + '/' + sample + '.jpg', faces_img[0])
    else:
        print('Image has more than one face. Skipped!')
