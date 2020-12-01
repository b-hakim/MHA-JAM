import numpy as np
import glob


def load_data(states_filepath, context_filesdir):
    with open(states_filepath) as fr:
        agents_states = fr.readlines()

    # format
    # agent_id, 20x(frame_id, x, y, v, a, yaw_rate)]
    agents_states = np.array([[float(x.rstrip()) for x in s.split(',')] for s in agents_states])
    contexts_files = glob.glob(context_filesdir+"*.npy")
    keys = [int(cf.split("trainval")[1].split(".txt")[0])for cf in contexts_files]
    agents_context = np.zeros((len(keys), 50, 50, 8, 5), dtype="float16")
    contexts_files = list(zip(contexts_files, keys))
    contexts_files.sort(key=lambda x: x[1])
    contexts_files = list(zip(*contexts_files))[0]

    for i, cf in enumerate(contexts_files):
        agents_context[i] = np.load(cf).reshape(50, 50, 8, 5)

    agents_states_x, agents_states_y = agents_states[:, :1+8*6], agents_states[:, 1+8*6:]
    return agents_states_x[:,1:].reshape(-1, 8, 6), agents_states_y.reshape(-1, 12, 6), agents_context

