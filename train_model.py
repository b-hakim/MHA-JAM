import argparse
import os

import tensorflow as tf

from tqdm import tqdm
from data_loader import load_data, load_map_batch
from model import build_model_mha_jam, build_model_mha_sam
from enum import Enum

import numpy as np


class MODEL_TYPE(Enum):
    MHA_JAM = 0
    MHA_SAM = 1


# Transform train_on_batch return value
# to dict expected by on_batch_end callback
def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = float(l[1])
    return result


def run(pdd, mode, model_type, model_save_dir, save_best_model_only, epochs, batch_size):
    train_states, train_context, train_map_dir, val_states, val_context, val_map_dir = os.path.join(pdd, "states_train_" + mode + ".txt"), \
                                                                                       os.path.join(pdd, "context_train_" + mode + "/"), \
                                                                                       os.path.join(pdd, "maps_train_" + mode + "/"), \
                                                                                       os.path.join(pdd, "states_val_" + mode + ".txt"), \
                                                                                       os.path.join(pdd, "context_val_" + mode + "/"), \
                                                                                       os.path.join(pdd, "maps_val_" + mode + "/")

    train_states_x, train_states_y, train_context_x = load_data(train_states, train_context)
    # train_states_x, train_states_y, train_context_x = train_states_x[:1], train_states_y[:1], train_context_x[:1]

    if model_type.value == MODEL_TYPE.MHA_JAM.value:
        model = build_model_mha_jam()
    else:
        model = build_model_mha_sam()

    # model.load_weights("model_iterations/model_mha_best.h5")
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=0,
        batch_size=batch_size,
        write_graph=True,
        write_grads=True
    )
    tensorboard.set_model(model)

    batches = int(np.ceil(train_states_x.shape[0]/batch_size))
    least_loss = -1
    loss_diverging = 0


    for e in range(0,epochs):
        losses_sum = 0

        for i in tqdm(range(batches)):
            start = i*batch_size
            end = start + batch_size

            if i == batches - 1:
                end = len(train_states_x)

            train_map_x = load_map_batch(train_map_dir, start, end)

            loss = model.train_on_batch([train_states_x[start:end,:,1:]*100, train_context_x[start:end]*100, train_map_x],
                                        train_states_y[start:end,:,1:3].reshape(-1, 12, 2)*100, return_dict=True)

            # predictions = model.predict([train_states_x[start:end,:,1:], train_context_x[start:end], train_map_x], verbose=1)

            # print(predictions)

            losses_sum += loss['loss']

            tensorboard.on_epoch_end(int(i+e*batch_size), loss)

        with open("output.txt", 'a') as fw:
            fw.writelines(["Epoch " + str(e) + ": " + str(losses_sum/batches)+"\n"])

        print("\nEpoch", str(e) + ":", losses_sum)

        # predictions = model.predict([val_states_x[:, :, 1:], val_context_x[start:end]], verbose=1)
        # ade, fde = calculate_ade_fde(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 2))
        # print(ade, fde)

        # with open("output.txt", 'a') as fw:
        #     fw.writelines(["ade, fde: " + str(ade) + ", " + str(fde)])

        if losses_sum/batches < least_loss or least_loss == -1:
            loss_diverging = 0
            least_loss = losses_sum/batches

            if save_best_model_only:
                model.save(os.path.join(model_save_dir, "model_mha_best.h5"), save_format="tf", include_optimizer=True)
            else:
                model.save(os.path.join(model_save_dir, "model_mha_"+str(e)+".h5"), save_format="tf", include_optimizer=True)
        else:
            loss_diverging += 1

        if loss_diverging == 500:
            break


if __name__ == '__main__':
    # mode = "v1.0-trainval"
    # mode = "v1.0-mini"

    parser = argparse.ArgumentParser(description='Generate the required files from dataset')
    parser.add_argument('--mode', type=str, default='v1.0-mini')
    parser.add_argument('--preprocessed_dataset_dir', type=str, default='/home/bassel/repos/nuscenes/mha-jam')
    parser.add_argument('--model_type', type=str, default='JAM')
    parser.add_argument('--model_save_dir', type=str, default='/mnt/23f8bdba-87e9-4b65-b3f8-dd1f9979402e/model_iterations_lstm')
    parser.add_argument('--save_best_model_only', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()

    if args.model_type == 'JAM':
        model_type = MODEL_TYPE.MHA_JAM
    else:
        model_type = MODEL_TYPE.MHA_SAM

    run(args.preprocessed_dataset_dir, args.mode, model_type, args.model_save_dir,
        args.save_best_model_only, args.epochs, args.batch_size)

'''
lines to change for using gaussian into euclidean and vise versa:
                                            - Model.py: 175 >> loss
                                            -           159 >> repeated vector
                                            -           150 >> Dense to be 4 or 2 respectively
                                            - evaluate_model.py >> 125 calculating error
                                            -                   >> 99 loading the model 
'''