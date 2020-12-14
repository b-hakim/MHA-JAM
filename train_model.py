from tqdm import tqdm
from data_loader import load_data, load_map_batch
from evaluate_model import calculate_ade_fde
from model import build_model


def run():
    mini = False

    if not mini:
        mode = "trainval"
        train_states, train_context, train_map_dir, val_states, val_context, val_map_dir = "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_train_v1.0-trainval.txt", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_train_v1.0-trainval/", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/maps_train_v1.0-trainval/", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_val_v1.0-trainval.txt", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_val_v1.0-trainval/", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/maps_val_v1.0-trainval/"
    else:
        mode = "mini"
        train_states, train_context, train_map_dir, val_states, val_context, val_map_dir = "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_train_v1.0-mini.txt", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_train_v1.0-mini/", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/maps_train_v1.0-mini", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_val_v1.0-mini.txt", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_val_v1.0-mini/", \
                                                               "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/maps_val_v1.0-mini/"

    # train_agents_ids_dict_path = "/media/bassel/Future/Study/Ph.D/Courses/CISC873/nuscenes-devkit/dicts_sample_and_instances_id2token_train.json"
    train_states_x, train_states_y, train_context_x = load_data(train_states, train_context)

    # val_agents_ids_dict_path = "/media/bassel/Future/Study/Ph.D/Courses/CISC873/nuscenes-devkit/dicts_sample_and_instances_id2token_val.json"
    # val_states_x, val_states_y, val_context_x, val_map_x = load_data(val_states, val_context, val_map)

    model = build_model()
    BATCH_SIZE=8
    EPOCHS=30
    batches = train_states_x.shape[0]//BATCH_SIZE

    for e in range(EPOCHS):
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
    run()
