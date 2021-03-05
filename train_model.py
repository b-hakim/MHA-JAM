from tqdm import tqdm
from data_loader import load_data, load_map_batch
from evaluate_model import calculate_ade_fde
from model import build_model_mha_jam, build_model_mha_sam
from enum import Enum


class MODEL_TYPE(Enum):
    MHA_JAM = 0
    MHA_SAM = 1


def run(pdd, mode, model_type=MODEL_TYPE.MHA_JAM):
    train_states, train_context, train_map_dir, val_states, val_context, val_map_dir = pdd + "states_train_" + mode + ".txt", \
                                                                                       pdd + "context_train_" + mode + "/", \
                                                                                       pdd + "maps_train_" + mode + "/", \
                                                                                       pdd + "states_val_" + mode + ".txt", \
                                                                                       pdd + "context_val_" + mode + "/", \
                                                                                       pdd + "maps_val_" + mode + "/"

    train_states_x, train_states_y, train_context_x = load_data(train_states, train_context)

    if model_type.value == MODEL_TYPE.MHA_JAM.value:
        model = build_model_mha_jam()
    else:
        model = build_model_mha_sam()

    model.load_weights("model_iterations/model_mha_9.h5")

    BATCH_SIZE=4
    EPOCHS=110
    batches = train_states_x.shape[0]//BATCH_SIZE

    for e in range(10,EPOCHS):
        losses_sum = 0

        for i in tqdm(range(batches)):
            start = i*BATCH_SIZE
            end = start + BATCH_SIZE

            if i == batches - 1:
                end = len(train_states_x)

            train_map_x = load_map_batch(train_map_dir, start, end)

            loss = model.train_on_batch([train_states_x[start:end,:,1:], train_context_x[start:end], train_map_x],
                                        train_states_y[start:end,:,1:3], return_dict=True)
            losses_sum += loss['loss']

        with open("output.txt", 'a') as fw:
            fw.writelines(["Epoch " + str(e) + ": " + str(losses_sum/batches)+"\n"])

        print("\nEpoch", str(e) + ":", losses_sum)

        # predictions = model.predict([val_states_x[:, :, 1:], val_context_x[start:end]], verbose=1)
        # ade, fde = calculate_ade_fde(val_states_y[:, :, 1:3], predictions.reshape(-1, 12, 2))
        # print(ade, fde)

        # with open("output.txt", 'a') as fw:
        #     fw.writelines(["ade, fde: " + str(ade) + ", " + str(fde)])

        model.save("model_iterations/model_mha_"+str(e)+".h5", save_format="tf")


if __name__ == '__main__':
    mode = "v1.0-trainval"
    # mode = "v1.0-mini"
    model_type = MODEL_TYPE.MHA_JAM
    preprocessed_dataset_dir = "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/"

    run(preprocessed_dataset_dir, mode, model_type)
