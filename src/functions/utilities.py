#!/usr/bin/python3

import os
import numpy as np
import math
import cv2

# open pose
JOINTS_POSE = [0, 15, 16, 17, 18]
# hip, neck, right and left shoulder
JOINTS_TRACKING = [8, 1, 2, 5]
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
DISTANCE_THRESHOLD = 50
THRESHOLD_HISTORY_TRACKING = 10


def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    if mean_x >= IMAGE_WIDTH:
        mean_x = IMAGE_WIDTH-1
    if mean_x < 0:
        mean_x = 0
    if mean_y >= IMAGE_HEIGHT:
        mean_y = IMAGE_HEIGHT-1
    if mean_y < 0:
        mean_y = 0

    return [mean_x, mean_y]


def joint_set(p):
    return p[0] != 0.0 or p[1] != 0.0


def dist_2d(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)

    squared_dist = np.sum((p1 - p2)**2, axis=0)
    return np.sqrt(squared_dist)


def get_openpose_bbox(pose):

    n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint])]
    if n_joints_set:
        centroid = compute_centroid(n_joints_set)

        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.2
        max_x += (max_x - min_x) * 0.2

        width = max_x - min_x

        min_y = centroid[1] - (width/3)*2
        max_y = centroid[1] + (width/3)*2

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        #print("Joint set empty!")
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
            #else:
            #    print("empty bbox! skipped")

    # ordered bboxes by min_x
    if bboxes:
        bboxes_array = np.asarray(bboxes)
        ordered_index = bboxes_array[:, 0].argsort(kind='mergesort')
        bboxes = [bboxes[i] for i in ordered_index]

        # extract face from images
        for bbox in bboxes:
            face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            resized = cv2.resize(face, required_size, interpolation=cv2.INTER_AREA)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            faces.append(resized)

    return faces, bboxes, ordered_index


def filter_faces(human_depth, faces_img, bboxes, order, num_selected_faces):
    depths = []

    new_faces_img = []
    new_bboxes = []
    new_order = []

    for bbox in bboxes:
        center_bbox = [bbox[0]+((bbox[2]-bbox[0])/2), bbox[1]+((bbox[3]-bbox[1])/2)]
        depths.append(get_mean_depth_over_area(human_depth, center_bbox, 2))

    for i in range(0, num_selected_faces):
        min_depth = min(depths)
        min_index = depths.index(min_depth)

        new_faces_img.append(faces_img[min_index])
        new_bboxes.append(bboxes[min_index])
        new_order.append(order[min_index])

        depths.pop(min_index)

    return new_faces_img, new_bboxes, new_order


def get_closer_poses(human_depth, poses, conf_poses, faces, conf_faces, distance):
    new_poses = []
    new_conf_poses = []
    new_faces = []
    new_conf_faces = []

    for idx, pose in enumerate(poses):
        n_joints_set = [pose[joint] for joint in JOINTS_POSE if joint_set(pose[joint])]
        if n_joints_set:
            depth_joints = []
            for joint in n_joints_set:
                if int(joint[1]) < 0:
                    joint[1] = 0
                elif int(joint[1]) >= IMAGE_HEIGHT:
                    joint[1] = IMAGE_HEIGHT-1
                if int(joint[0]) < 0:
                    joint[0] = 0
                elif int(joint[0]) >= IMAGE_WIDTH:
                    joint[0] = IMAGE_WIDTH-1

                depth_joint = human_depth[int(joint[1]), int(joint[0])]
                if depth_joint > 0:
                    depth_joints.append(depth_joint)

            if depth_joints:
                depth = np.median(depth_joints)

                if depth <= distance:
                    new_poses.append(pose)
                    new_conf_poses.append(conf_poses[idx])
                    new_faces.append(faces[idx])
                    new_conf_faces.append(conf_faces[idx])

    return new_poses, new_conf_poses, faces, new_conf_faces


def get_mean_depth_over_area(image_depth, pixel, range):

    vertical_range = np.zeros(2)
    vertical_range[0] = pixel[1] - round(range / 2) if pixel[1] - round(range / 2) > 0 else 0
    vertical_range[1] = pixel[1] + round(range / 2) if pixel[1] + round(range / 2) < IMAGE_HEIGHT else IMAGE_HEIGHT

    horizontal_range = np.zeros(2)
    horizontal_range[0] = pixel[0] - round(range / 2) if pixel[0] - round(range / 2) > 0 else 0
    horizontal_range[1] = pixel[0] + round(range / 2) if pixel[0] + round(range / 2) < IMAGE_WIDTH else IMAGE_WIDTH

    vertical_range = vertical_range.astype(int)
    horizontal_range = horizontal_range.astype(int)

    depth = []
    for hpix in np.arange(horizontal_range[0], horizontal_range[1]):
        for vpix in np.arange(vertical_range[0], vertical_range[1]):
            depth.append(image_depth[vpix, hpix])

    mean_depth = np.mean(depth)

    return mean_depth


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
                                face_part = [item.get(0).asFloat64(), item.get(1).asFloat64(), item.get(2).asFloat64()]
                                face_person.append(face_part)
                        else:
                            body_part = [part.get(1).asFloat64(), part.get(2).asFloat64(), part.get(3).asFloat64()]
                            body_person.append(body_part)

                if body_person:
                    body.append(body_person)
                if face_person:
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


def draw_bboxes(image, bboxes, color):

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image, start_point, end_point, color=color, thickness=2)

    return image


# get the face embedding for one face
def get_embedding(model, faces):
    # first face
    face = faces[0].astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    # transform face into one sample
    sample = np.expand_dims(face, axis=0)
    stack = np.vstack([sample])

    if len(faces) > 1:
        for i in range(1, len(faces)):
            # scale pixel values
            face = faces[i].astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face.mean(), face.std()
            face = (face - mean) / std
            # transform face into one sample
            sample = np.expand_dims(face, axis=0)
            stack = np.vstack([stack, sample])

    # make prediction to get embedding
    yhat = model.predict(stack)

    return yhat