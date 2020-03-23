import os
import pickle
import h5py
from collections import defaultdict

import numpy as np

from baselines import logger
from mpi4py import MPI

def is_square(n):
    return n == (int(np.sqrt(n))) ** 2


class Dataset(object):
    def __init__(self, filename):
        self.h5_episode = 0
        self.h5_handle = h5py.File(filename, 'a')
        self.create_group()

    def get_dataset(self, dataset):
        return self.current_grp.get(dataset)

    def create_group(self):
        self.current_grp = self.h5_handle.create_group("episode{}".format(str(self.h5_episode)))

    def add_to_dataset(self, dataset, arr):
        ds = self.get_dataset(dataset)
        if not ds:
            import pdb; pdb.set_trace()
        if ds.shape[0] == 1:
            ds.resize((ds.shape[0]-1) + arr.shape[0], axis=0)
        else:
            ds.resize(ds.shape[0] + arr.shape[0], axis=0)
        ds[-arr.shape[0]:] = arr

    def recreate(self, keys):
        del self.h5_handle['episode{}'.format(self.h5_episode)]
        # verify if the delete is successfull
        self.create_group()
        self.initialize(keys)

    def initialize(self, keys):
        for key in keys:
            if key not in list(self.current_grp.keys()):
                if key == 'obs':
                    self.current_grp.create_dataset(key, (1,84,84,4), dtype='f',
                                                    maxshape=(None,84,84,4), chunks=True)
                elif key == 'attention':
                    self.current_grp.create_dataset(key, (1,9,9,64), dtype='f',
                                                    maxshape=(None,9,9,64), chunks=True)
                elif key != 'acs':
                    self.current_grp.create_dataset(key, (1,), dtype='f',
                                                    maxshape=(None,), chunks=True)
                else:
                    self.current_grp.create_dataset(key, (1,), dtype='i',
                                                    maxshape=(None,), chunks=True)


class Recorder(object):
    def __init__(self, nenvs, score_multiple=1):
        self.episodes = [defaultdict(list) for _ in range(nenvs)]
        self.total_episodes = 0
        self.filename = self.get_filename()
        #self.dataset = Dataset(self.get_h5filename())
        self.score_multiple = score_multiple

        self.all_scores = {}
        self.all_places = {}

    def record(self, bufs, infos):
        #self.dataset.initialize(bufs)

        for env_id, ep_infos in enumerate(infos):
            left_step = 0
            done_steps = sorted(ep_infos.keys())
            for right_step in done_steps:
                for key in bufs:
                    self.episodes[env_id][key].append(bufs[key][env_id, left_step:right_step].copy())
                    #arr_copy = bufs[key][env_id, left_step:right_step].copy()
                    #self.dataset.add_to_dataset(key, arr_copy)

                self.record_episode(env_id, ep_infos[right_step])
                #self.record_episode(env_id, ep_infos[right_step], bufs.keys())
                left_step = right_step
                for key in bufs:
                    self.episodes[env_id][key].clear()
            for key in bufs:
                self.episodes[env_id][key].append(bufs[key][env_id, left_step:].copy())
                #self.dataset.add_to_dataset(key, arr_copy)

    #def record_episode(self, env_id, info, keys):
    def record_episode(self, env_id, info):
        self.total_episodes += 1
        if self.episode_worth_saving(env_id, info):
            episode = {}
            for key in self.episodes[env_id]:
                episode[key] = np.concatenate(self.episodes[env_id][key])
            info['env_id'] = env_id
            # info.pop("rng_at_episode_start",None)
            # for k,v in info.items():
            #     if isinstance(v, set):
            #         self.dataset.current_grp.attrs[k] = list(v)
            #     else:
            #         self.dataset.current_grp.attrs[k] = v
            # self.dataset.h5_episode += 1
            # self.dataset.create_group()
            # self.dataset.initialize(keys)

            episode['info'] = info
            with open(self.filename, 'ab') as f:
                pickle.dump(episode, f, protocol=-1)
        # else:
        #     self.dataset.recreate(keys)

    def get_score(self, info):
        return int(info['r']/self.score_multiple) * self.score_multiple

    def episode_worth_saving(self, env_id, info):
        if self.score_multiple is None:
            return False
        r = self.get_score(info)
        if r not in self.all_scores:
            self.all_scores[r] = 0
        else:
            self.all_scores[r] += 1
        hashable_places = tuple(sorted(info['places']))
        if hashable_places not in self.all_places:
            self.all_places[hashable_places] = 0
        else:
            self.all_places[hashable_places] += 1
        if is_square(self.all_scores[r]) or is_square(self.all_places[hashable_places]):
            return True
        if 15 in info['places']:
            return True
        return False

    def get_filename(self):
        filename = os.path.join(logger.get_dir(), 'videos_{}.pk'.format(MPI.COMM_WORLD.Get_rank()))
        return filename

    def get_h5filename(self):
        filename = os.path.join(logger.get_dir(), 'videos_{}.hdf5'.format(MPI.COMM_WORLD.Get_rank()))
        return filename

