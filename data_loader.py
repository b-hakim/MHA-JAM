import os

import numpy as np
import glob
import cv2
from tensorflow.keras.applications.vgg19 import preprocess_input


MAX_TRAJ_LEN = 28

def load_map_batch(map_files_dir, start, end, set="train"):
    # agents_map = np.zeros((end-start, 500, 500, 3))
    agents_map = np.zeros((end-start, 1024, 1024, 3), dtype='uint8')

    for i in range(start, end):
        mf = os.path.join(map_files_dir, "maps_"+set+f"__{i}.jpg")
        map_img = cv2.imread(mf)
        # agents_map[i-start] = cv2.resize(map_img, (500, 500))
        # agents_map[i-start] = map_img
        # agents_map[i-start] = np.zeros((1024, 1024, 3))
        # prepare the image for the VGG model
        # from BGR to RGB as it expects it in that format
        map_img = cv2.resize(map_img, (1024, 1024))
        agents_map[i-start] = preprocess_input(map_img[:,:,::-1])

    return agents_map


def load_data(states_filepath, context_filesdir):
    with open(states_filepath) as fr:
        agents_states = fr.readlines()

    ########################################################################
    # Context
    # format
    # agent_id, 28x(frame_id, x, y, v, a, yaw_rate)]
    agents_states = np.array([[float(x.rstrip()) for x in s.split(',')] for s in agents_states])
    # agents_states = agents_states.reshape(agents_states.shape)
    contexts_files = glob.glob(context_filesdir+"*.npy")
    for cf in contexts_files:
        int(cf.split("__")[1].split(".txt")[0])

    keys = [int(cf.split("__")[1].split(".txt")[0])for cf in contexts_files]

    agents_context = np.zeros((len(keys), 32, 32, MAX_TRAJ_LEN, 5), dtype='float16')
    contexts_files = list(zip(contexts_files, keys))
    contexts_files.sort(key=lambda x: x[1])
    contexts_files = list(zip(*contexts_files))[0]

    for i, cf in enumerate(contexts_files[:len(agents_states)]):
        agents_context[i] = np.load(cf).reshape(32, 32, MAX_TRAJ_LEN, 5)

    agents_states_x, agents_states_y = agents_states[:, :1+MAX_TRAJ_LEN*6], agents_states[:, 1+MAX_TRAJ_LEN*6:]
    # agents_context[agents_context == -64] = 0

    return agents_states_x[:,1:].reshape(-1, MAX_TRAJ_LEN, 6), agents_states_y.reshape(-1, 12, 6), agents_context

