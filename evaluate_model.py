import gc
import math

import scipy
import tensorflow.keras as k
import numpy as np
from tqdm import tqdm

from data_loader import load_data
from model import build_model, euclidean_distance_loss


# def distance_metrics(gt, preds):
#     errors = np.zeros(preds.shape[:-1])
#     for i in range(errors.shape[0]):
#         for j in range(errors.shape[1]):
#             errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
#     return errors.mean(), errors[:, -1].mean(), errors

# train the model
def calculate_ade_fde(actual, predicted):
    # calculate an ADE
    ade = 0

    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            sub = actual[row, col]-predicted[row, col]
            sum_sq = np.dot(sub, sub)
            ade += np.sqrt(sum_sq)

    ade /= actual.shape[0]*actual.shape[1]

    # calculate FDE
    fde = 0
    for row in range(actual.shape[0]):
        sub = actual[row, actual.shape[1]-1] - predicted[row, actual.shape[1]-1]
        sum_sq = np.dot(sub, sub)
        fde += np.sqrt(sum_sq)

    fde /= actual.shape[0]
    return ade, fde



def run():
    val_states, val_context = "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_val_v1.0-trainval.txt", \
                              "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_val/"

    val_states_x, val_states_y, val_context_x = load_data(val_states, val_context)

    # model = build_model()
    # model.load_weights()
    min_ade, min_fde = [100, -1], [100, -1]
    # for i in range(100):
    if True:
        i = 20
        model = k.models.load_model(f"model_iterations/model_mha_{i}.h5", custom_objects={'euclidean_distance_loss': euclidean_distance_loss})

        predictions = model.predict([val_states_x[:,:,1:], val_context_x[:]], verbose=1)
        ade, fde = calculate_ade_fde(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 2))

        if ade < min_ade[0]:
            min_ade = [ade, i]
        if fde < min_fde[0]:
            min_fde = [fde, i]

        print("Epoch", i, ":", ade, fde )
        k.backend.clear_session()
        gc.collect()
        del model

    print(min_ade, min_fde)

        # print(model.evaluate([val_states_x[start:end,:,1:], val_context_x[start:end]],
        #                val_states_y[start:end,:,1:3],))

if __name__ == '__main__':
    run()