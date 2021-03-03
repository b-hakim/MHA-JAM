import json
import math
import shutil
import cv2
import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import convert_global_coords_to_local
from tqdm import tqdm
from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

MAX_TRAJ_LEN = 28

class NuScenesFormatTransformer:
    def __init__(self, DATAROOT='./data/sets/nuscenes', dataset_version='v1.0-mini'):
        self.DATAROOT = DATAROOT
        self.dataset_version = dataset_version
        self.nuscenes = NuScenes(dataset_version, dataroot=self.DATAROOT)
        self.helper = PredictHelper(self.nuscenes)

    def get_format_mha_jam(self, samples_agents, out_file="./transformer_format.txt"):
        instance_token_to_id_dict = {}
        sample_token_to_id_dict = {}

        scene_token_dict = {}
        sample_id = 0
        instance_id = 0

        for current_sample in samples_agents:
            instance_token, sample_token = current_sample.split("_")
            scene_token = self.nuscenes.get('sample', sample_token)["scene_token"]

            if scene_token in scene_token_dict:
                continue

            # get the first sample in this sequence
            scene_token_dict[scene_token] = True
            first_sample_token = self.nuscenes.get("scene", scene_token)["first_sample_token"]
            current_sample = self.nuscenes.get('sample', first_sample_token)

            while True:
                if current_sample['token'] not in sample_token_to_id_dict:
                    sample_token_to_id_dict[current_sample['token']] = sample_id
                    sample_token_to_id_dict[sample_id] = current_sample['token']
                    sample_id += 1
                else:
                    print("should not happen?")

                instances_in_sample = self.helper.get_annotations_for_sample(current_sample['token'])

                for sample_instance in instances_in_sample:
                    if sample_instance['instance_token'] not in instance_token_to_id_dict:
                        instance_token_to_id_dict[sample_instance['instance_token']] = instance_id
                        instance_token_to_id_dict[instance_id] = sample_instance['instance_token']
                        instance_id += 1

                if current_sample['next'] == "":
                    break

                current_sample = self.nuscenes.get('sample', current_sample['next'])

        mode = "train" if out_file.find("_train") != -1 else "val"
        mini = "mini" if out_file.find("mini") != -1 else "main"

        with open("dicts_sample_and_instances_id2token_" + mode + "_" + mini + ".json", 'w') as fw:
            json.dump([instance_token_to_id_dict, sample_token_to_id_dict], fw)
        #############
        # Converting to the transformer network format
        # frame_id, agent_id, pos_x, pos_y
        # todo:
        # loop on all the agents, if agent not taken:
        # 1- add it to takens agents (do not retake the agent)
        # 2- get the number of appearance of this agent
        # 3- skip this agent if the number is less than 10s (4 + 6)
        # 4- get the middle agent's token
        # 5- get the past and future agent's locations relative to its location
        samples_new_format = []
        taken_instances = {}
        ds_size = 0
        # max_past_traj_len = -1

        for current_sample in samples_agents:
            instance_token, sample_token = current_sample.split("_")
            instance_id = instance_token_to_id_dict[instance_token]

            if instance_id in taken_instances:
                continue

            taken_instances[instance_id] = True

            # trajectory_full_instances = self.get_trajectory_around_sample(instance_token, sample_token,
            #                                                               just_xy=False)

            # //////////////////////
            future_samples = self.helper.get_future_for_agent(instance_token, sample_token, 6, True, False)
            past_samples = self.helper.get_past_for_agent(instance_token, sample_token, 1000, True, False)[::-1]

            current_sample = self.helper.get_sample_annotation(instance_token, sample_token)
            assert len(past_samples) >= 1
            assert len(future_samples) == 12

            # assert len(past_samples) < 7
            # if len(past_samples) > max_past_traj_len:
            #     max_past_traj_len = len(past_samples)

            # past_samples = np.append(past_samples, [current_sample], axis=0)

            ds_size += 1

            # get_trajectory at this position
            center_pos = len(past_samples)
            future_samples_local = self.helper.get_future_for_agent(instance_token, sample_token, 6, True, True)
            past_samples_local = self.helper.get_past_for_agent(instance_token, sample_token, 1000, True, True)[::-1]
            # current_sample = self.helper.get_sample_annotation(instance_token, sample_token)
            assert len(future_samples_local) == 12

            # if len(past_samples) > 7:
            #     past_samples = past_samples[len(past_samples)-7:]
            #     past_samples_local = past_samples_local[past_samples_local.shape[0]-7:]

            trajectory = np.append(past_samples_local, np.append([[0, 0]], future_samples_local, axis=0), axis=0)

            past_samples = [sample_token_to_id_dict[p['sample_token']] for p in past_samples]
            future_samples = [sample_token_to_id_dict[p['sample_token']] for p in future_samples]
            trajectory_tokens = np.append(past_samples, np.append([sample_token_to_id_dict[sample_token]], future_samples, axis=0), axis=0)

            trajectory_ = np.zeros((trajectory.shape[0], 6))
            trajectory_[:, 0] = trajectory_tokens[:]
            trajectory_[:, 1:3] = trajectory
            trajectory = trajectory_
            len_future_samples = len(future_samples)
            del trajectory_, trajectory_tokens, past_samples, future_samples, past_samples_local, future_samples_local

            curr_sample = self.helper.get_past_for_agent(instance_token, sample_token, 1000, False, False)[-1]

            for i in range(trajectory.shape[0]):
                # instance_id, sample_id, x, y, velocity, acc, yaw
                velocity = self.helper.get_velocity_for_agent(instance_token, curr_sample["sample_token"])
                acceleration = self.helper.get_acceleration_for_agent(instance_token, curr_sample["sample_token"])
                heading_change_rate = self.helper.get_heading_change_rate_for_agent(instance_token, curr_sample["sample_token"])

                if math.isnan(velocity):
                    velocity = 0
                if math.isnan(acceleration):
                    acceleration = 0
                if math.isnan(heading_change_rate):
                    heading_change_rate = 0

                # need to check paper for relative velocity? same for acc and yaw
                trajectory[i][3:] = [velocity, acceleration, heading_change_rate]
                # if curr_sample['next'] == '':
                #     import pdb
                #     pdb.set_trace()

                # No need to get next sample token in case this is last element in the series
                # prevents bug
                if i < trajectory.shape[0]-1:
                    next_sample_token = self.nuscenes.get('sample_annotation', curr_sample['next'])['sample_token']
                    curr_sample = self.helper.get_sample_annotation(instance_token, next_sample_token)

            s = str(instance_id) + ","
            assert (MAX_TRAJ_LEN+len_future_samples) >= trajectory.shape[0]
            repeat = (MAX_TRAJ_LEN+len_future_samples)-trajectory.shape[0]
            leading_arr = np.array(repeat * [-1, -64, -64, -64, -64, -64]).reshape((repeat, 6))
            trajectory = np.append(leading_arr, trajectory, axis=0)

            for i in range(trajectory.shape[0]):
                sample_id, x, y, velocity, acceleration, heading_change_rate = trajectory[i]
                s += str(sample_id) + "," + str(x) + "," + str(y) + "," + str(velocity) + "," \
                     + str(acceleration) + "," + str(heading_change_rate)
                if i != trajectory.shape[0]-1:
                    s += ","
                else:
                    s += "\n"

            samples_new_format.append(s)

        # print("max past trajectory len:",max_past_traj_len)

        # samples_new_format.sort(key=lambda x: int(x.split(",")[0]))

        with open(out_file, 'w') as fw:
            fw.writelines(samples_new_format)

        print(out_file + "size " + str(ds_size))

    def get_format_mha_jam_context(self, states_filepath, out_file):
        with open(states_filepath) as fr:
            agents_states = fr.readlines()

        # format
        # agent_id, 20x(frame_id, x, y, v, a, yaw_rate)]
        agents_states = [[float(x.rstrip()) for x in s.split(',')] for s in agents_states]

        mode = "train" if out_file.find("_train") != -1 else "val"
        mini = "mini" if out_file.find("mini") != -1 else "main"

        with open("dicts_sample_and_instances_id2token_" + mode + "_" + mini + ".json") as fr:
            instance_dict_id_token, sample_dict_id_token = json.load(fr)

        # Get Context for each sample in states
        context = []
        agent_ind = 0

        for agent in tqdm(agents_states):
            instance_token = instance_dict_id_token[str(int(agent[0]))]
            mid_frame_id = int(agent[1 + 6 * 7])
            sample_token = sample_dict_id_token[str(mid_frame_id)]
            frame_annotations = self.helper.get_annotations_for_sample(sample_token)
            surroundings_agents_coords = []
            surroundings_agents_instance_token = []

            for ann in frame_annotations:
                if ann['category_name'].find("vehicle") == -1:
                    continue
                if ann['instance_token'] == instance_token:
                    agent_ann = ann
                else:
                    surroundings_agents_coords.append(ann["translation"][:2])
                    surroundings_agents_instance_token.append(ann["instance_token"])

            if len(surroundings_agents_coords) != 0:
                surroundings_agents_coords = convert_global_coords_to_local(surroundings_agents_coords,
                                                                            agent_ann["translation"],
                                                                            agent_ann["rotation"])

            for i in range(len(surroundings_agents_coords)):
                if surroundings_agents_coords[i][0] < -25 or surroundings_agents_coords[i][0] > 25 \
                        or surroundings_agents_coords[i][1] < -10 or surroundings_agents_coords[i][1] > 40:
                    surroundings_agents_coords[i] = None
                    surroundings_agents_instance_token[i] = None

            total_area_side = 50
            cell_size = 1.5625
            map_side_size = int(total_area_side // cell_size)

            map = [[[-64, -64, -64, -64, -64] for i in range(MAX_TRAJ_LEN)] for j in range(map_side_size * map_side_size)]

            for n in range(len(surroundings_agents_coords)):
                if np.isnan(surroundings_agents_coords[n][0]) or surroundings_agents_coords[n] is None:
                    continue
                # search for the agent location in the map
                agent_found = False
                for i in range(map_side_size):
                    for j in range(map_side_size):
                        pos = i * map_side_size + j
                        # if agent found in the cell
                        if surroundings_agents_coords[n][0] >= (j * cell_size) - 25 \
                                and surroundings_agents_coords[n][0] < (j * cell_size) - 25 + cell_size \
                                and surroundings_agents_coords[n][1] >= i * cell_size - 10 \
                                and surroundings_agents_coords[n][1] < i * cell_size - 10 + cell_size:

                            past_trajectory = self.get_current_past_trajectory(surroundings_agents_instance_token[n],
                                                                               sample_token, num_seconds=1000)[:MAX_TRAJ_LEN]
                            assert len(past_trajectory) <= MAX_TRAJ_LEN
                            retrieved_trajectory_len = len(past_trajectory)

                            if map[pos][-1][0] != -64:
                                skip_traj = False
                                # Save the trajectory with greater length
                                for ind, map_pos in enumerate(map[pos]):
                                    if map_pos[0] != 64:
                                        if MAX_TRAJ_LEN - ind > retrieved_trajectory_len:
                                            skip_traj = True
                                if skip_traj:
                                    agent_found = True
                                    break
                                else:
                                    # print("new longer agent trajectory in cell")
                                    pass

                            past_trajectory = convert_global_coords_to_local(past_trajectory,
                                                                             agent_ann["translation"],
                                                                             agent_ann["rotation"])

                            if retrieved_trajectory_len != MAX_TRAJ_LEN:
                                past_trajectory = np.concatenate(
                                    [np.array([[-64, -64] for _ in range(MAX_TRAJ_LEN - past_trajectory.shape[0])]),
                                     past_trajectory], axis=0)

                            neighbour_agent_features = []

                            skip_traj = False

                            for k in range(0, MAX_TRAJ_LEN):
                                if retrieved_trajectory_len > k:
                                    if k == 0:
                                        sample_token_i = sample_dict_id_token[str(mid_frame_id)]
                                    else:
                                        sample_token_i = self.helper.get_sample_annotation(
                                            surroundings_agents_instance_token[n], sample_token_i)["prev"]
                                        sample_token_i = self.nuscenes.get('sample_annotation', sample_token_i)['sample_token']
                                    try:
                                        velocity = self.helper.get_velocity_for_agent(
                                            surroundings_agents_instance_token[n], sample_token_i)
                                    except:
                                        skip_traj = True
                                        # print("error")
                                        break
                                    acceleration = self.helper.get_acceleration_for_agent(
                                        surroundings_agents_instance_token[n],
                                        sample_token_i)
                                    heading_change_rate = self.helper.get_heading_change_rate_for_agent(
                                        surroundings_agents_instance_token[n],
                                        sample_token_i)
                                    if math.isnan(velocity):
                                        velocity = 0
                                    if math.isnan(acceleration):
                                        acceleration = 0
                                    if math.isnan(heading_change_rate):
                                        heading_change_rate = 0

                                    neighbour_agent_features.append([velocity, acceleration, heading_change_rate])
                                else:
                                    neighbour_agent_features.append([-64, -64, -64])

                            if skip_traj:
                                print("skipping agent because it has missing data")
                                agent_found = True
                                break

                            past_trajectory = np.concatenate([past_trajectory, neighbour_agent_features], axis=1)
                            map[pos] = past_trajectory.tolist()
                            agent_found = True
                            break
                    if agent_found:
                        break

            map = np.array(map).astype(np.float16)
            # context.append(map)
            np.save(out_file.replace("_.txt", "__" + str(agent_ind) + ".txt"), map)
            agent_ind += 1

            # with open(out_file, 'ab') as fw:
            #     pickle.dump(map, fw)
            #     continue
            # fw.write(map)

    def get_current_past_trajectory(self, instance_token, sample_token, num_seconds, just_xy=True,
                                    in_agent_frame=False):
        past_samples = self.helper.get_past_for_agent(instance_token, sample_token,
                                                      num_seconds, in_agent_frame, just_xy)[::-1] #[0:7][::-1]
        current_sample = self.helper.get_sample_annotation(instance_token, sample_token)

        if just_xy:
            current_sample = current_sample["translation"][:2]
            if past_samples.shape[0] == 0:
                trajectory = np.array([current_sample])
            else:
                trajectory = np.append(past_samples, [current_sample], axis=0)
        else:
            trajectory = np.append(past_samples, [current_sample], axis=0)
        return trajectory

    def get_format_mha_jam_maps(self, states_filepath, out_file):
        with open(states_filepath) as fr:
            agents_states = fr.readlines()

        # format
        # agent_id, 20x(frame_id, x, y, v, a, yaw_rate)]
        agents_states = [[float(x.rstrip()) for x in s.split(',')] for s in agents_states]

        mode = "train" if out_file.find("_train") != -1 else "val"
        mini = "mini" if out_file.find("mini") != -1 else "main"

        with open("dicts_sample_and_instances_id2token_" + mode + "_" + mini + ".json") as fr:
            instance_dict_id_token, sample_dict_id_token = json.load(fr)

        # Get map for each sample in states
        agent_ind = 0
        static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=1)
        mtp_input_representation = InputRepresentation(static_layer_rasterizer, agent_rasterizer, Rasterizer())

        for agent in tqdm(agents_states):
            instance_token = instance_dict_id_token[str(int(agent[0]))]
            mid_frame_id = int(agent[1 + 6 * 7])
            sample_token = sample_dict_id_token[str(mid_frame_id)]
            img = mtp_input_representation.make_input_representation(instance_token, sample_token)
            # img = cv2.resize(img, (1024, 1024))
            cv2.imwrite(out_file.replace("_.jpg", "__" + str(agent_ind) + ".jpg"), img)
            agent_ind += 1

    def run(self):
        if self.dataset_version.find("mini") != -1:
            train_agents = get_prediction_challenge_split("mini_train", dataroot=self.DATAROOT)
            val_agents = get_prediction_challenge_split("mini_val", dataroot=self.DATAROOT)
        else:
            train_agents = get_prediction_challenge_split("train", dataroot=self.DATAROOT)
            train_agents.extend(get_prediction_challenge_split("train_val", dataroot=self.DATAROOT))
            val_agents = get_prediction_challenge_split("val", dataroot=self.DATAROOT)
        # mx =-1
        # for  in train_agents:
        #     instance_token, sample_token = current_sample.split("_")
        #     past_samples_local = self.helper.get_past_for_agent(instance_token, sample_token, 100, True, True)[::-1]
        #     if len(past_samples_local) > mx:
        #         mx = len(past_samples_local)
        # print("max length of the past sequences for trainval is:",mx)
        # for instance_token, sample_token in train_agents:
        #     past_samples_local = self.helper.get_past_for_agent(instance_token, sample_token, 100, True, True)[::-1]
        #     if len(past_samples_local) > mx:
        #         mx = len(past_samples_local)
        # print("max length of the past sequence for val is:",mx)
        # return

        self.get_format_mha_jam(train_agents,
                                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_train_" + self.dataset_version + ".txt")
        # self.get_format_mha_jam_context(
        #     "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_train_" + self.dataset_version + ".txt",
        #     "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_train_" + self.dataset_version + "/context_train_.txt")
        # self.get_format_mha_jam_maps(
        #     "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_train_" + self.dataset_version + ".txt",
        #     "/media/bassel/Entertainment/maps_train_" + self.dataset_version + "/maps_train_.jpg")

        self.get_format_mha_jam(val_agents,
                                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_val_" + self.dataset_version + ".txt")
        # self.get_format_mha_jam_context(
        #     "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_val_" + self.dataset_version + ".txt",
        #     "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/context_val_" + self.dataset_version + "/context_val_.txt")
        # self.get_format_mha_jam_maps(
        #     "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/states_val_" + self.dataset_version + ".txt",
        #     "/media/bassel/Entertainment/maps_val_" + self.dataset_version + "/maps_val_.jpg")

if __name__ == '__main__':

    # copy = True # Is not needed anymore
    copy = False
    # mini = True
    mini = False
    base_name = "states"

    if copy:
        if mini:
            shutil.copy(
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/" + base_name + "_train_v1.0-mini.txt",
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/train/" + base_name + "_train.txt")
            shutil.copy(
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/" + base_name + "_val_v1.0-mini.txt",
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/val/" + base_name + "_val.txt")
            shutil.copy(
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/" + base_name + "_val_v1.0-mini.txt",
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/test/" + base_name + "_val.txt")
        else:
            shutil.copy(
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/" + base_name + "_train_v1.0-trainval.txt",
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/train/" + base_name + "_train.txt")
            shutil.copy(
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/" + base_name + "_val_v1.0-trainval.txt",
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/val/" + base_name + "_val.txt")
            shutil.copy(
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/bkup/" + base_name + "_val_v1.0-trainval.txt",
                "/home/bassel/PycharmProjects/Trajectory-Transformer/datasets/nuscenes/test/" + base_name + "_val.txt")
    else:
        if not mini:
            n = NuScenesFormatTransformer('/media/bassel/Entertainment/nuscenes/',
                                          'v1.0-trainval')
        else:
            n = NuScenesFormatTransformer('/media/bassel/Entertainment/nuscenes/',
                                          'v1.0-mini')
        n.run()