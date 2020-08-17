import os

import matplotlib.pyplot as plt
import numpy as np
import cv2

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image


class RetinaNetDetector:
    """please ensure that the image was read as bgr"""

    def __init__(self, weights_path, max_detections=25, score_threshold=0.5, backbone='resnet50'):
        self.max_detections = max_detections
        self.score_thresh = score_threshold
        self.model = models.load_model(weights_path, backbone_name=backbone)


    def detect(self, img):
        # preprocess image for network
        img = preprocess_image(img)
        img, scale = resize_image(img)
        boxes, scores, labels = self.model.predict_on_batch(img.reshape(1, *img.shape))
        boxes /= scale
        confidence_mask = scores[0, :] > self.score_thresh
        indices = np.where(confidence_mask)[0]
        # select those scores
        accepted_scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-accepted_scores)[:self.max_detections]
        # select detections
        image_boxes = boxes[0, indices[scores_sort], :]
        image_boxes = image_boxes.astype(int)
        return image_boxes



def draw_boxes(img, boxes, color):
    for box in boxes:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness=3)
    return img

if __name__ == '__main__':
    weights_path = '/media/adam/e46d6141-876f-4b0c-90da-9e9e217986f2/betzalel_personal/araplus/202008171624_pred_model_resnet50_csv_36.h5'
    save_path = '/home/adam/Desktop/betzalel_personal/keras-retinanet/images/araplus/'
    mode = 's'
    if mode == 'save':
        os.makedirs(save_path)
    detector = RetinaNetDetector(weights_path)
    image_file = open('/home/adam/Desktop/betzalel_personal/keras-retinanet/data/araplus/annotations/not_train_annotations.csv', 'r')

    annotations = image_file.readlines()
    annotations = [line.split(',') for line in annotations]
    files = [l[0] for l in annotations]
    annotations_dict = {f: [] for f in files}
    for line in annotations:
        f = line[0]
        box = [int(b) for b in line[1:-1]]
        annotations_dict[f].append(box)
    for k, v in annotations_dict.items():
        img = cv2.imread(k)
        predicted_boxes = detector.detect(img)
        img = draw_boxes(img, predicted_boxes, (0, 0, 255))
        img = draw_boxes(img, v, (0, 255, 0))
        if mode == 'save':
            cv2.imwrite(save_path + os.path.basename(k), img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.show()


