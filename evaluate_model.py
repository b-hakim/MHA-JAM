import gc
import math
import os

import scipy
import tensorflow.keras as k
import numpy as np
from tqdm import tqdm

from data_loader import load_data, load_map_batch
from model import euclidean_distance_loss, gaussian_nll


def distance_metrics(gt, preds):
    errors = np.zeros(preds.shape[:-1])
    for i in range(errors.shape[0]):
        for j in range(errors.shape[1]):
            errors[i, j] = scipy.spatial.distance.euclidean(gt[i, j], preds[i, j])
    return errors.mean(), errors[:, -1].mean(), errors

# train the model
def calculate_ade_fde_gaussian(actual, predicted):
    # calculate an ADE
    ade = 0

    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            predicted_samples = np.random.multivariate_normal(predicted[row, col, 0::2],
                                                              np.identity(2)*np.exp(predicted[row, col, 1::2]))
            # predicted_samples = predicted[row, col, 0::2]/np.exp(predicted[row, col, 1::2])
            sub = actual[row, col]- predicted_samples
            sum_sq = np.dot(sub, sub)
            ade += np.sqrt(sum_sq)

    ade /= actual.shape[0]*actual.shape[1]

    # calculate FDE
    fde = 0
    for row in range(actual.shape[0]):
        predicted_samples = np.random.multivariate_normal(predicted[row, col, 0::2],
                                                          np.identity(2) * predicted[row, col, 1::2])
        sub = actual[row, actual.shape[1]-1] - predicted_samples
        sum_sq = np.dot(sub, sub)
        fde += np.sqrt(sum_sq)

    fde /= actual.shape[0]
    return ade, fde


def calculate_ade_fde_euclidean(actual, predicted):
    # calculate an ADE
    ade = 0

    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            sub = actual[row, col]- predicted[row, col]
            sum_sq = np.dot(sub, sub)
            ade += np.sqrt(sum_sq)

    ade /= actual.shape[0]*actual.shape[1]

    # calculate FDE
    fde = 0
    for row in range(actual.shape[0]):
        sub = actual[row, actual.shape[1]-1] - predicted[row, col]
        sum_sq = np.dot(sub, sub)
        fde += np.sqrt(sum_sq)

    fde /= actual.shape[0]
    return ade, fde



def run(pdd, mode):
    val_states, val_context, val_map = os.path.join(pdd, "states_val_"+mode+".txt"), \
                                       os.path.join(pdd, "context_val_"+mode+"/"), \
                                       os.path.join(pdd, "maps_val_"+mode+"/")
    # val_states, val_context, val_map = os.path.join(pdd, "states_train_"+mode+".txt"), \
    #                                    os.path.join(pdd, "context_train_"+mode+"/"), \
    #                                    os.path.join(pdd, "maps_train_"+mode+"/")

    val_states_x, val_states_y, val_context_x = load_data(val_states, val_context)
    # val_states_x, val_states_y, val_context_x = val_states_x[:1], val_states_y[:1], val_context_x[:1]

    min_ade, min_fde = [100, -1], [100, -1]
    BATCH_SIZE=8

    if BATCH_SIZE > val_states_y.shape[0]:
        batches = 1
        BATCH_SIZE = val_states_y.shape[0]
    else:
        batches = val_states_y.shape[0]//BATCH_SIZE

    # for i in range(30, 31):
        # model = k.models.load_model(f"model_iterations/model_mha_{i}.h5",
        #                             custom_objects={'euclidean_distance_loss': euclidean_distance_loss,
        #                                             'gaussian_nll':gaussian_nll})
    model = k.models.load_model(f"model_iterations/model_mha_best.h5",
                                custom_objects={'euclidean_distance_loss': euclidean_distance_loss})
    # model = k.models.load_model(f"model_iterations/model_mha_best.h5",
    #                             custom_objects={'gaussian_nll':gaussian_nll})
    predictions = []

    for b in range(batches):
        start = b * BATCH_SIZE
        end = start + BATCH_SIZE

        if b == batches - 1:
            end = len(val_states_x)

        val_map_x = load_map_batch(val_map, start, end, 'val')
        # val_map_x = load_map_batch(val_map, start, end, 'train')

        batch_predictions = model.predict([val_states_x[start:end, :, 1:]*100,
                                           val_context_x[start:end]*100,
                                           val_map_x], verbose=1)

        if len(predictions) == 0:
            predictions = batch_predictions
        else:
            predictions = np.concatenate([predictions, batch_predictions])
        # break

    predictions = np.array(predictions)

    # ade, fde = calculate_ade_fde_gaussian(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 4))
    ade, fde = calculate_ade_fde_euclidean(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 2)/100)

    # if ade < min_ade[0]:
    #     min_ade = [ade, i]
    # if fde < min_fde[0]:
    #     min_fde = [fde, i]

    # print("Model", i, ":", ade, fde)
    # k.backend.clear_session()
    # gc.collect()
    # del model

    # print(min_ade, min_fde)
    print(ade, fde)
    # print(model.evaluate([val_states_x[start:end,:,1:], val_context_x[start:end]],
    #                val_states_y[start:end,:,1:3],))

if __name__ == '__main__':
    # mode = "v1.0-trainval"
    mode = "v1.0-mini"
    preprocessed_dataset_dir = "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup"

    run(preprocessed_dataset_dir, mode)