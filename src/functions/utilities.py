#!/usr/bin/python3

import os
import numpy as np
import math
import cv2

# open pose
JOINTS_POSE = [0, 15, 16, 17, 18]
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    return [mean_x, mean_y]


def joint_set(p):
    return p[0] != 0.0 or p[1] != 0.0


def get_openpose_bbox(pose):

    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint])]
    if n_joints_set:
        centroid = compute_centroid(n_joints_set)

        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.4
        max_x += (max_x - min_x) * 0.4

        width = max_x - min_x

        min_y = centroid[1] - (width/3)*2
        max_y = centroid[1] + (width/3)*2

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        print("Joint set empty!")
        return None, None, None, None


def extract_faces(frame, poses, required_size):
    bboxes = []
    faces = []
    ordered_index = []
    for pose in poses:
        min_x, min_y, max_x, max_y = get_openpose_bbox(pose)
        if min_x is not None and min_y is not None and max_x is not None and max_y is not None:
            if min_x != max_x and min_y != max_y:
                bboxes.append([math.floor(min_x), math.floor(min_y), math.floor(max_x), math.floor(max_y)])
            else:
                print("empty bbox! skipped")

    # ordered bboxes by min_x
    if bboxes:
        bboxes_array = np.asarray(bboxes)
        ordered_index = bboxes_array[:, 0].argsort(kind='mergesort')
        bboxes = [bboxes[i] for i in ordered_index]

        # extract face from images
        for bbox in bboxes:
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            resized = cv2.resize(face, required_size, interpolation=cv2.INTER_AREA)
            faces.append(resized)

    return faces, bboxes, ordered_index


def read_openpose_data(received_data):
    body = []
    face = []
    if received_data:
        received_data = received_data.get(0).asList()
        for i in range(0, received_data.size()):
            keypoints = received_data.get(i).asList()

            if keypoints:
                body_person = []
                face_person = []
                for y in range(0, keypoints.size()):
                    part = keypoints.get(y).asList()
                    if part:
                        if part.get(0).asString() == "Face":
                            for z in range(1, part.size()):
                                item = part.get(z).asList()
                                face_part = [item.get(0).asDouble(), item.get(1).asDouble(), item.get(2).asDouble()]
                                face_person.append(face_part)
                        else:
                            body_part = [part.get(1).asDouble(), part.get(2).asDouble(), part.get(3).asDouble()]
                            body_person.append(body_part)

                if body_person and face_person:
                    body.append(body_person)
                    face.append(face_person)

    poses, conf_poses = load_many_poses(body)
    faces, conf_faces = load_many_faces(face)

    return poses, conf_poses, faces, conf_faces


def load_many_poses(data):
    poses = []
    confidences = []

    for person in data:
        poses.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return poses, confidences


def load_many_faces(data):
    faces = []
    confidences = []

    for person in data:
        faces.append(np.array(person)[:, 0:2])
        confidences.append(np.array(person)[:, 2])

    return faces, confidences


def draw_bboxes(image, bboxes):

    color = (255, 0, 0)
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image, start_point, end_point, color=color, thickness=2)

    return image


# get the face embedding for one face
def get_embedding(model, face):
    # scale pixel values
    face = face.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face.mean(), face.std()
    face_pixels = (face - mean) / std
    # transform face into one sample
    samples = np.expand_dims(face, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)

    return yhat[0]
