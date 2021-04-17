import argparse
import gc
import math
import os
from glob import glob

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
            # np.random.normal(predicted[row, col, 0::2], np.exp(predicted[row, col, 1::2]))
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


def run(pdd, mode, model_path, batch_size):
    val_states, val_context, val_map = os.path.join(pdd, "states_val_"+mode+".txt"), \
                                       os.path.join(pdd, "context_val_"+mode+"/"), \
                                       os.path.join(pdd, "maps_val_"+mode+"/")
    # # experimenting one batch
    # val_states, val_context, val_map = os.path.join(pdd, "states_train_"+mode+".txt"), \
    #                                    os.path.join(pdd, "context_train_"+mode+"/"), \
    #                                    os.path.join(pdd, "maps_train_"+mode+"/")

    val_states_x, val_states_y, val_context_x = load_data(val_states, val_context)
    # experimenting one batch
    # val_states_x, val_states_y, val_context_x = val_states_x[:1], val_states_y[:1], val_context_x[:1]

    min_ade, min_fde = [100, -1], [100, -1]

    if batch_size > val_states_y.shape[0]:
        batches = 1
        batch_size = val_states_y.shape[0]
    else:
        batches = val_states_y.shape[0]//batch_size

    if os.path.isfile(model_path):
        count=1
    else:
        count = len(glob(model_path+"/*.h5"))
        available_models = glob(model_path+"/*.h5")

    for i in range(count):
        if os.path.isfile(model_path):
            model = k.models.load_model(model_path,
                                        custom_objects={'euclidean_distance_loss': euclidean_distance_loss,
                                                        'gaussian_nll':gaussian_nll})
        else:
            model = k.models.load_model(available_models[i],
                                        custom_objects={'euclidean_distance_loss': euclidean_distance_loss,
                                                        'gaussian_nll':gaussian_nll})
        predictions = []

        for b in range(batches):
            start = b * batch_size
            end = start + batch_size

            if b == batches - 1:
                end = len(val_states_x)

            val_map_x = load_map_batch(val_map, start, end, 'val')
            # val_map_x = load_map_batch(val_map, start, end, 'train') # trying to overfit one/mini sample

            batch_predictions = model.predict([val_states_x[start:end, :, 1:]*100,
                                               val_context_x[start:end]*100,
                                               val_map_x], verbose=1)

            if len(predictions) == 0:
                predictions = batch_predictions
            else:
                predictions = np.concatenate([predictions, batch_predictions])

        predictions = np.array(predictions)

        # ade, fde = calculate_ade_fde_gaussian(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 4))
        ade, fde = calculate_ade_fde_euclidean(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 2)/100)

        if ade < min_ade[0]:
            min_ade = [ade, i]
        if fde < min_fde[0]:
            min_fde = [fde, i]

        if not os.path.isfile(model_path):
            print("Model", i, ":", ade, fde)
        k.backend.clear_session()
        gc.collect()
        del model

    print(min_ade, min_fde)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate the model error rates')
    parser.add_argument('--mode', type=str, default='v1.0-mini') #v1.0-mini
    parser.add_argument('--preprocessed_dataset_dir', type=str, default='/home/bassel/repos/nuscenes/mha-jam')
    parser.add_argument('--model_path', type=str, default='/mnt/23f8bdba-87e9-4b65-b3f8-dd1f9979402e/model_iterations',
                        help="Path of a file for using a specific model or pass the path of the folder to "
                                    "experiment all models inside the folder")
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    run(args.preprocessed_dataset_dir, args.mode, args.model_path, args.batch_size)