
from gcpds.image_segmentation.losses import DiceCoefficient
from tensorflow.keras import backend as K

def cam_dice(roi, cam):
    return -DiceCoefficient()(roi,cam)

def cam_cumulative_relevance(roi, cam):
    intersection = K.sum(roi * cam, axis=[1,2])
    union = K.sum(cam,axis=[1,2])+ K.epsilon()
    return intersection/union

def masked_cumulative_relevance(roi, cam):
    intersection = K.sum(roi * cam, axis=[1,2])
    union = K.sum(roi,axis=[1,2])+ K.epsilon()
    return intersection/union