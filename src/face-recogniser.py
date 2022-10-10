#!/usr/bin/python3

import numpy as np
import os
import yarp
import sys
import pickle as pk
from keras.models import load_model
import cv2
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from functions.utilities import read_openpose_data, extract_faces, draw_bboxes
from functions.utilities import get_embedding, compute_centroid, joint_set
from functions.utilities import IMAGE_WIDTH, IMAGE_HEIGHT, JOINTS_POSE_FACE

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
        self.labels_set = (rf.find("face_labels").asString()).split("-")
        print('Labels set: [%s]' % ', '.join(map(str, self.labels_set)))

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

        # output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/facerecogniser/pred:o')
        print('{:s} opened'.format('/facerecogniser/pred:o'))

        return True

    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        elif command.get(0).asString() == 'train':
            self.TRAIN = 1
            # init the models again
            self.dataset = []
            self.svm_model = None
            self.encoder = None
            self.normaliser = None
            self.name_file = command.get(1).asString()
            self.face_selected = command.get(2).asString()
            reply.addString('Training started. Files will be saved as ' + self.name_file + '. Face of interest: ' + self.face_selected)
        elif command.get(0).asString() == 'init':
            self.TRAIN = 0
            # init the models again
            self.dataset = []
            self.svm_model = None
            self.encoder = None
            self.normaliser = None
            reply.addString('Init done.')
        elif command.get(0).asString() == 'run':
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
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_prediction.close()
        return True

    def getPeriod(self):
        return 0.001

    def updateModule(self):

        received_image = self.in_port_human_image.read()

        if received_image:
            self.in_buf_human_image.copy(received_image)
            human_image = np.copy(self.in_buf_human_array)

            received_data = self.in_port_human_data.read()
            if received_data:
                try:
                    poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                    # images 160x160 pixels
                    # faces are ordered from left to right (x-axis increasing)
                    if poses and len(poses) <= len(self.labels_set):
                        faces_img, bboxes, order = extract_faces(human_image, poses, required_size=(160, 160))
                        for i in range(0, len(faces_img)):
                            cv2.imshow('image id %d' % i, cv2.cvtColor(faces_img[i], cv2.COLOR_BGR2RGB))
                            cv2.waitKey(1)
                        # print bounding box
                        human_image = draw_bboxes(human_image, bboxes)

                        # if TRAINING collect the dataset
                        if self.TRAIN == 1 and len(poses) == len(self.labels_set):
                            # suppose the sequence of people from left to right is always the same
                            # idx 0: therapist, id 1: child
                                for i in range(0, len(faces_img)):
                                    self.dataset.append((get_embedding(self.facenet_model, faces_img[i]), self.labels_set[i]))

                                # train each 50*2 samples (first time I'll get 100*2)
                                if len(self.dataset) > 100 and len(self.dataset) % 100 == 0:
                                    trainX = np.squeeze(np.asarray([data[0] for data in self.dataset[:-100]]))
                                    trainy = np.asarray([data[1] for data in self.dataset[:-100]])
                                    testX = np.squeeze(np.asarray([data[0] for data in self.dataset[-100:]]))
                                    testy = np.asarray([data[1] for data in self.dataset[-100:]])

                                    in_encoder = Normalizer(norm='l2')
                                    trainX = in_encoder.transform(trainX)
                                    testX = in_encoder.transform(testX)

                                    out_encoder = LabelEncoder()
                                    out_encoder.fit(trainy)
                                    trainy = out_encoder.transform(trainy)
                                    testy = out_encoder.transform(testy)

                                    base_model = SVC(kernel='rbf', probability=True, class_weight="balanced")
                                    base_model.fit(trainX, trainy)
                                    model = CalibratedClassifierCV(base_estimator=base_model, cv="prefit")
                                    model.fit(trainX, trainy)

                                    yhat_train = model.predict(trainX)
                                    yhat_test = model.predict(testX)

                                    # score
                                    score_train = accuracy_score(trainy, yhat_train)
                                    score_test = accuracy_score(testy, yhat_test)
                                    print('Dataset length: %d ; Accuracy: train=%.3f - test=%.3f' % (len(self.dataset), score_train, score_test))

                                    self.svm_model = model
                                    self.normaliser = in_encoder
                                    self.encoder = out_encoder

                                    if len(self.dataset) > 300*2 and score_train > 0.99 and score_test > 0.99:
                                        pk.dump(self.svm_model, open(self.output_path_models + 'svm_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.encoder, open(self.output_path_models + 'label_encoder_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.normaliser, open(self.output_path_models + 'normaliser_model_' + self.name_file + '.pkl', 'wb'))
                                        pk.dump(self.dataset, open(self.output_path_datasets + 'dataset_' + self.name_file + '.pkl', 'wb'))
                                        print("Training done. Models have been saved.")
                                        self.TRAIN = 0

                        # in the init phase everything is none
                        if self.svm_model is not None and self.encoder is not None and self.normaliser is not None:
                            # prediction for the face
                            data = []
                            for i in range(0, len(faces_img)):
                                data.append(get_embedding(self.facenet_model, faces_img[i]))

                            data = self.normaliser.transform(data)
                            data_array = np.asarray(data)
                            yhat_class = self.svm_model.predict_proba(data_array)
                            for itP in range(0, yhat_class.shape[0]):
                                # get name
                                prob = max(yhat_class[itP])
                                y_pred = (np.where(yhat_class[itP] == prob))[0]

                                predicted_name = self.encoder.inverse_transform(y_pred)

                                txt = "%s" % predicted_name[0]
                                human_image = cv2.putText(human_image, txt, tuple([int(bboxes[itP][0]), int(bboxes[itP][3])+20]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                                txt = "c: %0.1f" % prob
                                human_image = cv2.putText(human_image, txt, tuple([int(bboxes[itP][0]), int(bboxes[itP][3])+50]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                            if self.TRAIN == 0:
                                # send in output the selected pose from openpose as bottle
                                face_selected_idx = self.encoder.transform(np.asarray([self.face_selected]))
                                choice_idx = np.argmax(yhat_class[face_selected_idx, :])
                                openpose_idx = (int)(order[choice_idx])
                                pred = yarp.Bottle()
                                pred.addList().read((received_data.get(0).asList()).get(openpose_idx))
                                pred_list = yarp.Bottle()
                                pred_list.addList().read(pred)
                                self.out_port_prediction.write(pred_list)

                                selected_pose = poses[openpose_idx]
                                centroid = compute_centroid([selected_pose[joint] for joint in JOINTS_POSE_FACE if joint_set(selected_pose[joint])])
                                human_image = cv2.circle(human_image, tuple([int(centroid[0]), int(centroid[1])]), 6, (0, 0, 255), -1)
                    else:
                        print("Warning! Human faces detected > labels. Human faces: " + str(len(poses)))
                except Exception as err:
                    print("Unexpected error!!! " + err)

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

