#!/usr/bin/python3
import random

import numpy as np
import os
import yarp
import sys
import pickle as pk
from keras.models import load_model
import cv2
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import Normalizer, LabelEncoder
import imgaug.augmenters as iaa
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from functions.utilities import read_openpose_data, extract_faces, filter_faces, draw_bboxes
from functions.utilities import get_embedding, compute_centroid, joint_set
from functions.utilities import IMAGE_WIDTH, IMAGE_HEIGHT, JOINTS_POSE

yarp.Network.init()


class FaceRecogniser(yarp.RFModule):

    def configure(self, rf):
        self.TRAIN = 0
        model_name = rf.find("facenet_model_name").asString()
        self.facenet_model = load_model('./functions/' + model_name)
        print('Facenet model name: %s' % model_name)
        self.output_path_models = rf.find("output_path_models").asString()
        print('Output path for the models: %s' % self.output_path_models)
        self.output_path_datasets = rf.find("output_path_datasets").asString()
        print('Output path for the datasets: %s' % self.output_path_datasets)
        # labels must be divided by a dash, format accepted in ini file: pippo-pluto-paperino
        self.labels_set = (rf.find("face_labels").asString()).split("-")
        print('Labels set: [%s]' % ', '.join(map(str, self.labels_set)))
        self.path_lfw_dataset = rf.find("path_lfw_dataset").asString()
        print('Path LFW dataset: %s' % self.path_lfw_dataset)

        if not os.path.exists(self.output_path_models):
            os.makedirs(self.output_path_models)
        if not os.path.exists(self.output_path_datasets):
            os.makedirs(self.output_path_datasets)

        self.dataset = []
        self.svm_model = None
        self.encoder = None
        self.normaliser = None
        self.name_file = ""
        self.face_selected = ""

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

        # input port for depth
        self.in_port_human_depth = yarp.BufferedPortImageFloat()
        self.in_port_human_depth_name = '/facerecogniser/depth:i'
        self.in_port_human_depth.open(self.in_port_human_depth_name)
        self.in_buf_human_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.in_buf_human_depth = yarp.ImageFloat()
        self.in_buf_human_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_depth.setExternal(self.in_buf_human_depth_array.data, self.in_buf_human_depth_array.shape[1],
                                            self.in_buf_human_depth_array.shape[0])
        print('{:s} opened'.format('/facerecogniser/depth:i'))

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

        # output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/facerecogniser/pred:o')
        print('{:s} opened'.format('/facerecogniser/pred:o'))

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            # command: quit
            self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        elif command.get(0).asString() == 'train':
            # command: train namefile FOI --> train pippomodel pippo
            self.TRAIN = 1
            # init the models again to be sure
            self.dataset = []
            self.svm_model = None
            self.encoder = None
            self.normaliser = None
            self.name_file = command.get(1).asString()
            self.face_selected = command.get(2).asString()
            reply.addString('Training started. Files will be saved as ' + self.name_file + '. Face of interest: ' + self.face_selected)
        elif command.get(0).asString() == 'init':
            # command: init
            self.TRAIN = 0
            # init the models again
            self.dataset = []
            self.svm_model = None
            self.encoder = None
            self.normaliser = None
            reply.addString('Init done.')
        elif command.get(0).asString() == 'run':
            # command: run namefile FOI --> run pippomodel pippo
            self.TRAIN = 0
            # load svm model
            self.name_file = command.get(1).asString()
            self.svm_model = pk.load(open(self.output_path_models + 'svm_model_' + self.name_file + '.pkl', 'rb'))
            self.encoder = pk.load(open(self.output_path_models + 'label_encoder_model_' + self.name_file + '.pkl', 'rb'))
            self.normaliser = pk.load(open(self.output_path_models + 'normaliser_model_' + self.name_file + '.pkl', 'rb'))
            self.face_selected = command.get(2).asString()
            reply.addString('Run. Loaded the models ' + self.name_file + '. Face of interest: ' + self.face_selected)
        return True

    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):

        received_image = self.in_port_human_image.read()
        received_depth = self.in_port_human_depth.read()

        if received_image and received_depth:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)

            self.in_buf_human_depth.copy(received_depth)
            human_depth = np.copy(self.in_buf_human_depth_array)

            received_data = self.in_port_human_data.read()
            if received_data:
                try:
                    poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                    if poses:
                        # images 160x160 pixels
                        # faces are ordered from left to right (x-axis increasing)
                        faces_img, bboxes, order = extract_faces(human_image, poses, required_size=(160, 160))
                        if len(bboxes) > 0:
                            # filter bboxes by depth
                            # suppose that the N labelled faces are the closer
                            closer_faces_img, closer_bboxes, closer_order = filter_faces(human_depth, faces_img, bboxes, order, len(self.labels_set))
                            for i in range(0, len(closer_faces_img)):
                                cv2.imshow('image id %d' % i, closer_faces_img[i])
                                cv2.waitKey(1)
                            # print bounding box
                            human_image = draw_bboxes(human_image, bboxes, color=(0, 0, 255))
                            human_image = draw_bboxes(human_image, closer_bboxes, color=(255, 0, 0))

                            # if TRAINING collect the dataset
                            if self.TRAIN == 1:
                                # suppose the sequence of people from left to right is always the same and labelled
                                raw_data = []
                                for i in range(0, len(closer_faces_img)):
                                    raw_data.append((closer_faces_img[i], self.labels_set[i]))
                                    aug = iaa.Sequential([iaa.Fliplr(0.8), iaa.Rotate(rotate=(-45, 45)), iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))]).augment_image(closer_faces_img[i])
                                    raw_data.append((aug, self.labels_set[i]))
                                    cv2.imshow('image id %d' % i, aug)
                                    cv2.waitKey(1)
                                # append to the dataset also N samples from LFW
                                for i in range(len(self.labels_set), len(self.labels_set)+9):
                                    lfw_img = random.choice(self.lfw_dataset)
                                    raw_data.append((lfw_img, 'unknown'))
                                    lfw_img_aug = iaa.Sequential([iaa.Fliplr(0.8), iaa.Rotate(rotate=(-45, 45)), iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30))]).augment_image(lfw_img)
                                    raw_data.append((lfw_img_aug, 'unknown'))

                                embeds = get_embedding(self.facenet_model, [item[0] for item in raw_data])
                                [self.dataset.append((embeds[i], (raw_data[i])[1])) for i in range(0, len(embeds))]

                                # train each num_classes*100 samples
                                num_classes = (len(self.labels_set)+1)
                                if len(self.dataset) > 500*num_classes and len(self.dataset) % 1000 == 0:
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
                                    #base_model = SVC(kernel='rbf', probability=True)
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
                                    print('---------------')

                                    self.svm_model = model
                                    self.normaliser = in_encoder
                                    self.encoder = out_encoder

                                    if len(self.dataset) > 500*num_classes and score_test >= 0.99:
                                        pk.dump(self.svm_model, open(self.output_path_models + 'svm_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.encoder, open(self.output_path_models + 'label_encoder_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.normaliser, open(self.output_path_models + 'normaliser_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.dataset, open(self.output_path_datasets + 'dataset_' + self.name_file + '.pkl', 'wb'))
                                        print("Training done. Models have been saved.")
                                        self.TRAIN = 0

                            # in the init phase everything is none
                            if self.svm_model is not None and self.encoder is not None and self.normaliser is not None:
                                # prediction for the face
                                data = get_embedding(self.facenet_model, faces_img)
                                data = self.normaliser.transform(data)
                                yhat_class = self.svm_model.predict_proba(data)
                                y_preds = []
                                for itP in range(0, yhat_class.shape[0]):
                                    # get name
                                    prob = max(yhat_class[itP])
                                    y_pred = (np.where(yhat_class[itP] == prob))[0]
                                    y_preds.append((y_pred[0], prob))

                                    predicted_name = self.encoder.inverse_transform(y_pred)

                                    txt = "%s" % predicted_name[0]
                                    human_image = cv2.putText(human_image, txt, tuple([int(bboxes[itP][0]), int(bboxes[itP][3])+20]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                    txt = "c: %0.1f" % prob
                                    human_image = cv2.putText(human_image, txt, tuple([int(bboxes[itP][0]), int(bboxes[itP][3])+50]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                                if self.TRAIN == 0:
                                    # send in output the selected pose from openpose as bottle
                                    y_preds = np.array(y_preds)
                                    face_selected_idx = self.encoder.transform(np.asarray([self.face_selected]))
                                    if face_selected_idx in y_preds[:, 0]:
                                        conf_max = np.max((y_preds[y_preds[:, 0] == face_selected_idx])[:, 1])
                                        choice_idx = [idx for idx in range(0, y_preds.shape[0]) if y_preds[idx, 0] == face_selected_idx and y_preds[idx, 1] == conf_max]
                                        openpose_idx = (int)(order[choice_idx])
                                        selected_pose = poses[openpose_idx]
                                        centroid = compute_centroid(
                                            [selected_pose[joint] for joint in JOINTS_POSE if joint_set(selected_pose[joint])])

                                        if centroid is not None and not np.isnan(np.array(centroid)).all():
                                            pred = yarp.Bottle()
                                            pred.addList().read((received_data.get(0).asList()).get(openpose_idx))
                                            pred_list = yarp.Bottle()
                                            pred_list.addList().read(pred)
                                            self.out_port_prediction.write(pred_list)

                                            human_image = cv2.circle(human_image, tuple([int(centroid[0]), int(centroid[1])]), 6, (255, 0, 0), -1)
                        else:
                            print('Cannot extract any face from OpenPose')
                    else:
                        print('No human detected by OpenPose')
                except Exception as err:
                    print("Unexpected error!!! " + str(err))

            txt = "TRAINING: %d, NAME: %s" % (self.TRAIN, self.name_file)
            human_image = cv2.putText(human_image, txt, tuple([20, 450]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            # write rgb image
            self.out_buf_human_array[:, :] = human_image
            self.out_port_human_image.write(self.out_buf_human_image)

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

