#!/usr/bin/python3

import tensorflow as tf
import random, datetime
import numpy as np
import statistics
import os
import yarp
import sys
import pickle as pk
from keras.models import load_model
import cv2
import glob
import distutils.util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import imgaug.augmenters as iaa

from functions.utilities import read_openpose_data, extract_faces, draw_bboxes, get_closer_poses_bbox, filter_faces_bbox
from functions.utilities import get_embedding, compute_centroid, joint_set, dist_2d
from functions.utilities import IMAGE_WIDTH, IMAGE_HEIGHT, JOINTS_POSE_FACE, JOINTS_TRACKING, THRESHOLD_HISTORY_TRACKING, DISTANCE_THRESHOLD


yarp.Network.init()


def init_gpus(num_gpu, num_gpu_start):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if num_gpu > 0:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpus[num_gpu_start:num_gpu_start+num_gpu], 'GPU')
                print(len(gpus), "Physical GPUs found, Set visible devices: ", tf.config.experimental.get_visible_devices('GPU'))
            else:
                tf.config.experimental.set_visible_devices([], 'GPU')
                print("GPU has been disable!!!")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No physical GPU found")


class FaceRecogniser(yarp.RFModule):

    def configure(self, rf):
        num_gpu = rf.find("num_gpu").asInt32()
        num_gpu_start = rf.find("num_gpu_start").asInt32()
        print('Num GPU: %d, GPU start: %d' % (num_gpu, num_gpu_start))
        init_gpus(num_gpu, num_gpu_start)

        self.TRAIN = 0
        model_name = rf.find("facenet_model_name").asString()
        self.facenet_model = load_model('./models/' + model_name)
        print('Facenet model name: %s' % model_name)
        self.output_path_models = './models/' #rf.find("output_path_models").asString()
        print('Output path for the models: %s' % self.output_path_models)
        self.output_path_datasets = './models/' #rf.find("output_path_datasets").asString()
        print('Output path for the datasets: %s' % self.output_path_datasets)
        # labels must be divided by a dash, format accepted in ini file: pippo-pluto-paperino
        self.labels_set = (rf.find("face_labels").asString()).split("-")
        print('Labels set: [%s]' % ', '.join(map(str, self.labels_set)))
        self.path_lfw_dataset = '../LFW/' #rf.find("path_lfw_dataset").asString()
        print('Path LFW dataset: %s' % self.path_lfw_dataset)
        self.HUMAN_TRACKING = bool(distutils.util.strtobool((rf.find("human_tracking").asString())))
        print('Human tracking (hip): %s' % str(self.HUMAN_TRACKING))
        self.HUMAN_DISTANCE_THRESHOLD = rf.find("bbox_threshold").asInt32()
        print('Bbox diagonal threshold for the recognition: %d pixels' % self.HUMAN_DISTANCE_THRESHOLD)
        self.MIN_SIZE_DATASET = rf.find("min_size_dataset").asInt32()
        print('Min size for the dataset: %d samples' % self.MIN_SIZE_DATASET)
        self.ACCURACY_THRESHOLD = rf.find("accuracy_threshold").asFloat64()
        print('Accuracy threshold to stop the training: %.2f ' % self.ACCURACY_THRESHOLD)

        self.dataset = []
        self.svm_model = None
        self.encoder = None
        self.normaliser = None
        self.name_file = ""
        self.face_selected = ""
        if self.HUMAN_TRACKING:
            self.prev_human_joints_tracking = [None]*len(JOINTS_TRACKING)
            self.threshold_history_tracking_count = 0

        # init lfw dataset
        self.lfw_dataset = []
        imgfiles = glob.glob(os.path.join(self.path_lfw_dataset, '*.jpg'))
        for imgfile in imgfiles:
            img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
            self.lfw_dataset.append(img)

        # command port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/facerecogniser/command:i')
        print('{:s} opened'.format('/facerecogniser/command:i'))
        self.attach(self.cmd_port)

        # input port for rgb image
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/facerecogniser/image:i')
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1],
                                            self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/facerecogniser/image:i'))

        # input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/facerecogniser/data:i')
        print('{:s} opened'.format('/facerecogniser/data:i'))

        # output port for bboxes
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/facerecogniser/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1],
                                             self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/facerecogniser/image:o'))

        # propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/facerecogniser/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/facerecogniser/propag:o'))

        # output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/facerecogniser/pred:o')
        print('{:s} opened'.format('/facerecogniser/pred:o'))

        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        elif command.get(0).asString() == 'train':
            # command: train namefile FOI --> train pippomodel pippo
            # init the models again
            self.TRAIN = 1
            self.dataset = []
            self.svm_model = None
            self.encoder = None
            self.normaliser = None
            self.name_file = command.get(1).asString()
            self.face_selected = command.get(2).asString()
            if self.HUMAN_TRACKING:
                self.prev_human_joints_tracking = [None] * len(JOINTS_TRACKING)
                self.threshold_history_tracking_count = 0
            print("Training command received at ", datetime.datetime.now())
            reply.addString('Training started. Files will be saved as ' + self.name_file + '. Face of interest: ' + self.face_selected)
        elif command.get(0).asString() == 'run':
            # command: run namefile FOI --> run pippomodel pippo
            # load svm model
            self.name_file = command.get(1).asString()
            self.svm_model = pk.load(open(self.output_path_models + 'svm_model_' + self.name_file + '.pkl', 'rb'))
            self.encoder = pk.load(open(self.output_path_models + 'label_encoder_model_' + self.name_file + '.pkl', 'rb'))
            self.normaliser = pk.load(open(self.output_path_models + 'normaliser_model_' + self.name_file + '.pkl', 'rb'))
            self.face_selected = command.get(2).asString()
            self.TRAIN = 0
            reply.addString('Run. Loaded the models ' + self.name_file + '. Face of interest: ' + self.face_selected)
        elif command.get(0).asString() == 'init':
            # command: init
            self.TRAIN = 0
            # init the models and the rest again
            self.dataset = []
            self.svm_model = None
            self.encoder = None
            self.normaliser = None
            self.name_file = ""
            self.face_selected = ""
            if self.HUMAN_TRACKING:
                self.prev_human_joints_tracking = [None] * len(JOINTS_TRACKING)
                self.threshold_history_tracking_count = 0
            reply.addString('Init done.')
        return True

    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):

        received_image = self.in_port_human_image.read()

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)
            self.human_image = np.copy(human_image)

            received_data = self.in_port_human_data.read()
            if received_data:
                #try:
                    poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                    if poses:
                        received_data, poses, conf_poses, faces, conf_faces = get_closer_poses_bbox(received_data, poses, conf_poses, faces, conf_faces, self.HUMAN_DISTANCE_THRESHOLD)
                    # images 160x160 pixels
                    # faces are ordered from left to right (x-axis increasing)
                    if poses:
                        faces_img, bboxes, order = extract_faces(human_image, poses, required_size=(160, 160))
                        if len(bboxes) > 0:
                            # print bounding box
                            human_image = draw_bboxes(human_image, bboxes, color=(0, 0, 255))

                            # if TRAINING collect the dataset
                            if self.TRAIN == 1:
                                closer_faces_img, closer_bboxes, closer_order = filter_faces_bbox(faces_img, bboxes, order, len(self.labels_set))
                                human_image = draw_bboxes(human_image, closer_bboxes, color=(255, 0, 0))

                                # suppose the sequence of people from left to right is always the same and labelled
                                raw_data = []
                                for i in range(0, len(closer_faces_img)):
                                    raw_data.append((closer_faces_img[i], self.labels_set[i]))
                                    aug = iaa.Sequential([iaa.Fliplr(0.8), iaa.Rotate(rotate=(-45, 45)), iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))]).augment_image(closer_faces_img[i])
                                    raw_data.append((aug, self.labels_set[i]))
                                # append to the dataset also N samples from LFW
                                for i in range(len(self.labels_set), len(self.labels_set) + 9):
                                    lfw_img = random.choice(self.lfw_dataset)
                                    raw_data.append((lfw_img, 'unknown'))
                                    lfw_img_aug = iaa.Sequential([iaa.Fliplr(0.8), iaa.Rotate(rotate=(-45, 45)), iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))]).augment_image(lfw_img)
                                    raw_data.append((lfw_img_aug, 'unknown'))

                                embeds = get_embedding(self.facenet_model, [item[0] for item in raw_data])
                                [self.dataset.append((embeds[i], (raw_data[i])[1])) for i in range(0, len(embeds))]

                                # train each num_classes*100 samples
                                num_classes = (len(self.labels_set) + 1)
                                if len(self.dataset) >= 50*num_classes and len(self.dataset) % 100 == 0:
                                    datasetX = np.asarray([data[0] for data in self.dataset])
                                    datasetY = np.asarray([data[1] for data in self.dataset])
                                    trainX, testX, trainy, testy = train_test_split(datasetX, datasetY, test_size=0.3)
                                    print('Dataset length: %d' % len(self.dataset))

                                    in_encoder = Normalizer(norm='l2')
                                    trainX = in_encoder.transform(trainX)
                                    testX = in_encoder.transform(testX)

                                    out_encoder = LabelEncoder()
                                    out_encoder.fit(trainy)
                                    trainy = out_encoder.transform(trainy)
                                    testy = out_encoder.transform(testy)

                                    params = {'C': np.linspace(1, 30, 50), 'gamma': np.linspace(0.0001, 1, 50)}
                                    base_model = RandomizedSearchCV(SVC(kernel='rbf', probability=True), param_distributions=params)
                                    # base_model = SVC(kernel='rbf', probability=True)
                                    base_model.fit(trainX, trainy)
                                    model = CalibratedClassifierCV(base_estimator=base_model, cv="prefit")
                                    model.fit(trainX, trainy)
                                    print("The best parameters are %s with a score of %0.2f" % (base_model.best_params_, base_model.best_score_))

                                    yhat_train = model.predict(trainX)
                                    yhat_test = model.predict(testX)

                                    # score
                                    score_train = accuracy_score(trainy, yhat_train)
                                    score_test = accuracy_score(testy, yhat_test)
                                    print('Accuracy: train=%.3f - test=%.3f' % (score_train, score_test))
                                    print('Precision: test=%.3f ' % precision_score(testy, yhat_test))
                                    print('Recall: test=%.3f ' % recall_score(testy, yhat_test))
                                    print('F1 score: test=%.3f ' % f1_score(testy, yhat_test))
                                    print('Timestamp: ', datetime.datetime.now())
                                    print('---------------')

                                    self.svm_model = model
                                    self.normaliser = in_encoder
                                    self.encoder = out_encoder

                                    #if len(self.dataset) > 200*num_classes and score_train > 0.9 and score_test > 0.8:
                                    if len(self.dataset) >= self.MIN_SIZE_DATASET * num_classes and score_test > self.ACCURACY_THRESHOLD:
                                        pk.dump(self.svm_model, open(self.output_path_models + 'svm_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.encoder, open(self.output_path_models + 'label_encoder_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.normaliser, open(self.output_path_models + 'normaliser_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.dataset, open(self.output_path_datasets + 'dataset_' + self.name_file + '.pkl', 'wb'))
                                        print("Training done. Models have been saved at ", datetime.datetime.now())
                                        self.TRAIN = 0

                            # in the init phase everything is none
                            if self.svm_model is not None and self.encoder is not None and self.normaliser is not None:
                                # prediction for the face
                                data = get_embedding(self.facenet_model, faces_img)
                                # if call train, the svm, normaliser and the encoder can be None online
                                if self.normaliser is not None and self.svm_model is not None:
                                    data = self.normaliser.transform(data)
                                    yhat_class = self.svm_model.predict_proba(data)
                                    y_preds = []
                                    for itP in range(0, yhat_class.shape[0]):
                                        # get name
                                        prob = max(yhat_class[itP])
                                        y_pred = (np.where(yhat_class[itP] == prob))[0]
                                        y_preds.append((y_pred[0], prob))

                                    # there is only one instance for each label in label_set
                                    if self.encoder is not None:
                                        for label in self.labels_set:
                                            max_conf = 0
                                            max_conf_idx = 0
                                            for itP in range(0, len(y_preds)):
                                                if y_preds[itP][0] == self.encoder.transform([label])[0]:
                                                    if y_preds[itP][1] > max_conf:
                                                        max_conf = y_preds[itP][1]
                                                        max_conf_idx = itP
                                            # keep the teacher only at idx
                                            for itP in range(0, len(y_preds)):
                                                if y_preds[itP][0] == self.encoder.transform([label])[0] and itP != max_conf_idx:
                                                    prob_temp = y_preds[itP][1]
                                                    y_preds[itP] = (self.encoder.transform(['unknown'])[0], prob_temp)

                                        for itP in range(0, len(y_preds)):
                                            predicted_name = self.encoder.inverse_transform([y_preds[itP][0]])

                                            txt = "%s" % predicted_name[0]
                                            human_image = cv2.putText(human_image, txt, tuple([int(bboxes[itP][0]), int(bboxes[itP][3]) + 20]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                            txt = "c: %0.1f" % y_preds[itP][1]
                                            human_image = cv2.putText(human_image, txt, tuple([int(bboxes[itP][0]), int(bboxes[itP][3]) + 50]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                                        if self.TRAIN == 0 and self.encoder is not None:
                                            # send in output the selected pose from openpose as bottle
                                            y_preds = np.array(y_preds)
                                            face_selected_idx = self.encoder.transform(np.asarray([self.face_selected]))
                                            if face_selected_idx in y_preds[:, 0]:
                                                conf_max = np.max((y_preds[y_preds[:, 0] == face_selected_idx])[:, 1])
                                                choice_idx = [idx for idx in range(0, y_preds.shape[0]) if y_preds[idx, 0] == face_selected_idx and y_preds[idx, 1] == conf_max]
                                                openpose_idx = int(order[choice_idx])
                                                selected_pose = poses[openpose_idx]
                                                centroid = compute_centroid(
                                                    [selected_pose[joint] for joint in JOINTS_POSE_FACE if joint_set(selected_pose[joint])])

                                                if centroid is not None and not np.isnan(np.array(centroid)).all():
                                                    pred = yarp.Bottle()
                                                    pred.addList().read((received_data.get(0).asList()).get(openpose_idx))
                                                    pred_list = yarp.Bottle()
                                                    pred_list.addList().read(pred)
                                                    self.out_port_prediction.write(pred_list)

                                                    human_image = cv2.circle(human_image, tuple([int(centroid[0]), int(centroid[1])]), 6, (255, 0, 0), -1)
                                                else:
                                                    if self.HUMAN_TRACKING:
                                                        print("Recognised the human but the face is not visible. Forward the recognised skeleton anyway.")
                                                        pred = yarp.Bottle()
                                                        pred.addList().read((received_data.get(0).asList()).get(openpose_idx))
                                                        pred_list = yarp.Bottle()
                                                        pred_list.addList().read(pred)
                                                        self.out_port_prediction.write(pred_list)

                                                        for joint in JOINTS_TRACKING:
                                                            if joint_set(selected_pose[joint]):
                                                                human_image = cv2.circle(human_image, tuple([int(selected_pose[joint][0]), int(selected_pose[joint][1])]), 6,(255, 0, 0), -1)

                                                if self.HUMAN_TRACKING:
                                                    for i in range(0, len(JOINTS_TRACKING)):
                                                        if joint_set(selected_pose[JOINTS_TRACKING[i]]):
                                                            self.prev_human_joints_tracking[i] = tuple(selected_pose[JOINTS_TRACKING[i]])
                                                        else:
                                                            self.prev_human_joints_tracking[i] = None

                                                    # human recognised, put the threshold on the history to zero
                                                    self.threshold_history_tracking_count = 0

                                            else:
                                                if self.HUMAN_TRACKING and self.threshold_history_tracking_count <= THRESHOLD_HISTORY_TRACKING:
                                                    print("People not recognised in the scene. Forwarding the track of the prev skeleton.")
                                                    # joints of all people in the scene
                                                    current_joints_poses_tracking = []
                                                    for pose in poses:
                                                        # joints for the single pose
                                                        current_joints_pose = []
                                                        for i in range(0, len(JOINTS_TRACKING)):
                                                            if joint_set(pose[JOINTS_TRACKING[i]]):
                                                                current_joints_pose.append(tuple(pose[JOINTS_TRACKING[i]]))
                                                            else:
                                                                current_joints_pose.append(None)
                                                        current_joints_poses_tracking.append(current_joints_pose)

                                                    dist = []
                                                    for row in current_joints_poses_tracking:
                                                        dist_row = []
                                                        for i in range(0, len(row)):
                                                            if row[i] is not None and self.prev_human_joints_tracking[i] is not None:
                                                                dist_row.append(dist_2d(row[i], self.prev_human_joints_tracking[i]))
                                                            else:
                                                                dist_row.append(None)
                                                        dist.append(dist_row)

                                                    # if len(dist) != 0 and not np.isnan(np.array(dist)).all():
                                                    if len(dist) != 0:
                                                        mean_dist = []
                                                        for row in dist:
                                                            if not all(v is None for v in row):
                                                                dist_without_None = [v for v in row if v is not None]
                                                                mean_dist.append(statistics.mean(dist_without_None))
                                                            else:
                                                                mean_dist.append(None)

                                                        if not all(v is None for v in mean_dist):
                                                            min_value = min(i for i in mean_dist if i is not None)
                                                            min_idx = mean_dist.index(min_value)
                                                            if min_value <= DISTANCE_THRESHOLD:
                                                                pred = yarp.Bottle()
                                                                pred.addList().read((received_data.get(0).asList()).get(min_idx))
                                                                pred_list = yarp.Bottle()
                                                                pred_list.addList().read(pred)
                                                                self.out_port_prediction.write(pred_list)

                                                                for joint in current_joints_poses_tracking[min_idx]:
                                                                    if joint is not None:
                                                                        human_image = cv2.circle(human_image, tuple([int(joint[0]), int(joint[1])]), 6,(255, 0, 0), -1)

                                                                self.prev_human_joints_tracking = current_joints_poses_tracking[min_idx]
                                                            else:
                                                                print('Closest human too far from the tracked one')
                                                    else:
                                                        print('Cannot track the skeleton at minimum distance')

                                                    self.threshold_history_tracking_count = self.threshold_history_tracking_count + 1
                            else:
                                # draw bbox of the closest one
                                closer_faces_img, closer_bboxes, closer_order = filter_faces_bbox(faces_img, bboxes, order, len(self.labels_set))
                                human_image = draw_bboxes(human_image, closer_bboxes, color=(255, 0, 0))

                                # openpose_idx = int(closer_order[0])
                                # selected_pose = poses[openpose_idx]
                                # centroid = compute_centroid(
                                #     [selected_pose[joint] for joint in JOINTS_POSE_FACE if joint_set(selected_pose[joint])])
                                #
                                # if centroid is not None and not np.isnan(np.array(centroid)).all():
                                #     pred = yarp.Bottle()
                                #     pred.addList().read((received_data.get(0).asList()).get(openpose_idx))
                                #     pred_list = yarp.Bottle()
                                #     pred_list.addList().read(pred)
                                #     self.out_port_prediction.write(pred_list)
                                #
                                #     human_image = cv2.circle(human_image, tuple([int(centroid[0]), int(centroid[1])]), 6, (255, 0, 0), -1)

                        else:
                            print('Skeleton detected but cannot extract any face from OpenPose')
                            # if the face is not visible
                            if self.HUMAN_TRACKING and self.threshold_history_tracking_count <= THRESHOLD_HISTORY_TRACKING:
                                # joints of all people in the scene
                                current_joints_poses_tracking = []
                                for pose in poses:
                                    # joints for the single pose
                                    current_joints_pose = []
                                    for i in range(0, len(JOINTS_TRACKING)):
                                        if joint_set(pose[JOINTS_TRACKING[i]]):
                                            current_joints_pose.append(tuple(pose[JOINTS_TRACKING[i]]))
                                        else:
                                            current_joints_pose.append(None)
                                    current_joints_poses_tracking.append(current_joints_pose)

                                dist = []
                                for row in current_joints_poses_tracking:
                                    dist_row = []
                                    for i in range(0, len(row)):
                                        if row[i] is not None and self.prev_human_joints_tracking[i] is not None:
                                            dist_row.append(dist_2d(row[i], self.prev_human_joints_tracking[i]))
                                        else:
                                            dist_row.append(None)
                                    dist.append(dist_row)

                                # if len(dist) != 0 and not np.isnan(np.array(dist)).all():
                                if len(dist) != 0:
                                    mean_dist = []
                                    for row in dist:
                                        if not all(v is None for v in row):
                                            dist_without_None = [v for v in row if v is not None]
                                            mean_dist.append(statistics.mean(dist_without_None))
                                        else:
                                            mean_dist.append(None)

                                    if not all(v is None for v in mean_dist):
                                        min_value = min(i for i in mean_dist if i is not None)
                                        min_idx = mean_dist.index(min_value)
                                        if min_value <= DISTANCE_THRESHOLD:
                                            pred = yarp.Bottle()
                                            pred.addList().read((received_data.get(0).asList()).get(min_idx))
                                            pred_list = yarp.Bottle()
                                            pred_list.addList().read(pred)
                                            self.out_port_prediction.write(pred_list)

                                            for joint in current_joints_poses_tracking[min_idx]:
                                                if joint is not None:
                                                    human_image = cv2.circle(human_image, tuple([int(joint[0]), int(joint[1])]), 6,(255, 0, 0), -1)

                                            self.prev_human_joints_tracking = current_joints_poses_tracking[min_idx]
                                        else:
                                            print('Closest human too far from the tracked one')
                                else:
                                    print('Cannot track the skeleton at minimum distance')

                                self.threshold_history_tracking_count = self.threshold_history_tracking_count + 1
                    else:
                        print('No human detected by OpenPose')
                #except Exception as err:
                        #print("Unexpected error!!! " + str(err))

            txt = "TRAINING: %d, NAME: %s" % (self.TRAIN, self.name_file)
            human_image = cv2.putText(human_image, txt, tuple([20, 450]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # write rgb image
            self.out_buf_human_array[:, :] = human_image
            self.out_port_human_image.write(self.out_buf_human_image)
            # propag received image
            self.out_buf_propag_image_array[:, :] = self.human_image
            self.out_port_propag_image.write(self.out_buf_propag_image)

        return True


if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("FaceRecogniser")
    rf.setDefaultConfigFile('../app/config/facerecogniser_conf.ini')

    rf.configure(sys.argv)

    # Run module
    manager = FaceRecogniser()
    manager.runModule(rf)

