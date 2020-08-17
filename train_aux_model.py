# in this script i will load my trained retinanet, freeze everything except the regression submodel
# then i will modify to accept a new input which will be some sort of feature vector which should feed into regression submodel
# then i will want to retrain

import numpy as np
from keras import Model
from keras import layers
from keras.utils.vis_utils import plot_model

from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image

if __name__ == '__main__':
    weights_path = '/media/adam/e46d6141-876f-4b0c-90da-9e9e217986f2/betzalel_personal/araplus/202008171624/weights/resnet50_csv_36.h5'
    model = models.load_model(weights_path, backbone_name='resnet50')
    model_inputs = model.input
    model_outputs = model.outputs
    input_priors = layers.Input((None, 4))
    x = layers.Dense(4)(input_priors)
    x = layers.Concatenate()([x, model_outputs[0]])
    x = layers.Dense(4)(x)
    m = Model(model_inputs + [input_priors], [model_outputs[1], x])
    print()
    """
    regression_model = model.layers[202]
    regression_inputs = regression_model.inputs
    regression_outputs = regression_model.output
    new_inputs = layers.Input(shape=(4,))
    new_reg_model = Model(regression_inputs + [new_inputs], regression_outputs + [layers.Dense(4)(regression_model(regression_inputs))])
    print()
    """