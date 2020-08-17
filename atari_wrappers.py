import os
import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
from copy import copy
from baselines import logger

cv2.ocl.setUseOpenCL(False)

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return float(np.sign(reward))


class OldWarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class EgoFrame:
    def __init__(self):
        self.ego_h = 30
        self.ego_w = 51

class MontezumaEgoFrame(EgoFrame):
    def __init__(self):
        self.lower_color = np.array([199, 71, 71], dtype="uint8")
        self.upper_color = np.array([201, 73, 73], dtype="uint8")
        super(MontezumaEgoFrame, self).__init__()

    def find_character_in_frame(self, frame):
        mask = cv2.inRange(frame, self.lower_color, self.upper_color)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        pix_x, pix_y, _ = np.where(output > 0)
        if pix_x.size != 0:
            prev_pix_x = pix_x
            pix_x = pix_x[np.where(pix_x > 19)]
            pix_y = pix_y[-pix_x.size:]

            # If array is even then median doesn't exist in the array, because it's the average
            # between the middle twos
            try:
                # Very rarely a nan will be received here
                median_x = int(np.median(pix_x))
                while median_x not in pix_x:
                    median_x += 1

                median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
            except Exception as e:
                logger.error("Exception: {}".format(e))
                logger.error("Pixel x: {}".format(pix_x))
                logger.error("Pixel y: {}".format(pix_y))
                logger.error("Previous pixel x: {}".format(prev_pix_x))
                roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                return roi

        else:
            median_x = output.shape[0] // 2
            median_y = output.shape[1] // 2

        low_x = median_x-self.ego_h
        high_x = median_x+self.ego_h
        low_y = median_y-self.ego_w
        high_y = median_y+self.ego_w

        low_x = low_x if low_x > 0 else 0
        high_x = high_x if high_x < frame.shape[0] else frame.shape[0]
        low_y = low_y if low_y > 0 else 0
        high_y = high_y if high_y < frame.shape[1] else frame.shape[1]

        roi = frame[low_x:high_x, low_y:high_y]
        return roi


class GravitarEgoFrame(EgoFrame):
    def __init__(self):
        self.lower_color = np.array([98, 180, 215], dtype="uint8")
        self.upper_color = np.array([105, 186, 220], dtype="uint8")
        super(GravitarEgoFrame, self).__init__()

    def find_character_in_frame(self, frame):
        mask = cv2.inRange(frame, self.lower_color, self.upper_color)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        pix_x, pix_y, _ = np.where(output > 0)
        if pix_x.size != 0:
            pix_x = pix_x[np.where(pix_x > 23)]
        if pix_x.size != 0:
            # In this case, the agents lives are blue
            prev_pix_x = pix_x
            pix_y = pix_y[-pix_x.size:]

            # If array is even then median doesn't exist in the array, because it's the average
            # between the middle twos
            try:
                median_x = int(np.median(pix_x))
                while median_x not in pix_x:
                    median_x += 1

                median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
            except Exception as e:
                """
                The agent can transform into a sort of parachute, this are the color ranges
                This case can also happen as the agent dies it disappears from the screen
                """
                mask = cv2.inRange(frame,
                                   np.array([250, 181, 215], dtype="uint8"),
                                   np.array([254, 185, 219], dtype="uint8"))
                output = cv2.bitwise_and(frame, frame, mask=mask)

                pix_x, pix_y, _ = np.where(output > 0)
                if pix_x.size != 0:
                    try:
                        median_x = int(np.median(pix_x))
                        while median_x not in pix_x:
                            median_x += 1

                        median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
                    except Exception as e:
                        roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                        return roi
                else:
                    roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                    return roi

        else:
            """
            In this case, the agents lives are another color
            The agent can transform into a sort of parachute, this are the color ranges
            This case can also happen as the agent dies it disappears from the screen
            """
            mask = cv2.inRange(frame,
                               np.array([250, 181, 215], dtype="uint8"),
                               np.array([254, 185, 219], dtype="uint8"))
            output = cv2.bitwise_and(frame, frame, mask=mask)

            pix_x, pix_y, _ = np.where(output > 0)
            if pix_x.size != 0:
                try:
                    # Very rarely a nan will be received here
                    median_x = int(np.median(pix_x))
                    while median_x not in pix_x:
                        median_x += 1

                    median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
                except Exception as e:
                    roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                    return roi
            else:
                roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                return roi

        low_x = median_x-self.ego_h
        high_x = median_x+self.ego_h
        low_y = median_y-self.ego_w
        high_y = median_y+self.ego_w

        low_x = low_x if low_x > 0 else 0
        high_x = high_x if high_x < frame.shape[0] else frame.shape[0]
        low_y = low_y if low_y > 0 else 0
        high_y = high_y if high_y < frame.shape[1] else frame.shape[1]

        roi = frame[low_x:high_x, low_y:high_y]
        return roi


class PitfallEgoFrame(EgoFrame):
    def __init__(self):
        self.lower_color = np.array([226, 109, 109], dtype="uint8")
        self.upper_color = np.array([230, 114, 114], dtype="uint8")
        super(PitfallEgoFrame, self).__init__()

    def find_character_in_frame(self, frame):
        mask = cv2.inRange(frame, self.lower_color, self.upper_color)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        pix_x, pix_y, _ = np.where(output > 0)
        if pix_x.size != 0:
            # If array is even then median doesn't exist in the array, because it's the average
            # between the middle twos
            try:
                # Very rarely a nan will be received here
                median_x = int(np.median(pix_x))
                while median_x not in pix_x:
                    median_x += 1

                median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
            except Exception as e:
                roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                return roi

        else:
            # We try to find the agent green torso
            mask = cv2.inRange(frame,
                               np.array([90, 184, 90], dtype="uint8"),
                               np.array([94, 188, 94], dtype="uint8"))
            output = cv2.bitwise_and(frame, frame, mask=mask)

            pix_x, pix_y, _ = np.where(output > 0)
            if pix_x.size != 0:
                try:
                    # Very rarely a nan will be received here
                    median_x = int(np.median(pix_x))
                    while median_x not in pix_x:
                        median_x += 1

                    median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
                except Exception as e:
                    roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                    return roi

            else:
                # We try to find the legs
                mask = cv2.inRange(frame,
                                   np.array([51, 93, 22], dtype="uint8"),
                                   np.array([55, 97, 26], dtype="uint8"))
                output = cv2.bitwise_and(frame, frame, mask=mask)

                pix_x, pix_y, _ = np.where(output > 0)
                if pix_x.size != 0:
                    pix_x = pix_x[np.where(pix_x > 64)]
                if pix_x.size != 0:
                    pix_y = pix_y[-pix_x.size:]
                    try:
                        # Very rarely a nan will be received here
                        median_x = int(np.median(pix_x))
                        while median_x not in pix_x:
                            median_x += 1

                        median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
                    except Exception as e:
                        roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                        return roi
                else:
                    # The agent is dead
                    roi = np.zeros([self.ego_h, self.ego_w, 3], dtype=np.uint8)
                    return roi


        low_x = median_x-self.ego_h
        high_x = median_x+self.ego_h
        low_y = median_y-self.ego_w
        high_y = median_y+self.ego_w

        low_x = low_x if low_x > 0 else 0
        high_x = high_x if high_x < frame.shape[0] else frame.shape[0]
        low_y = low_y if low_y > 0 else 0
        high_y = high_y if high_y < frame.shape[1] else frame.shape[1]

        roi = frame[low_x:high_x, low_y:high_y]
        return roi


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84

        if env.unwrapped.spec.id == 'MontezumaRevengeNoFrameskip-v4':
            self.ego_game = MontezumaEgoFrame()
        elif env.unwrapped.spec.id == 'GravitarNoFrameskip-v4':
            self.ego_game = GravitarEgoFrame()
        elif env.unwrapped.spec.id == 'PitfallNoFrameskip-v4':
            self.ego_game = PitfallEgoFrame()
        else:
            raise Exception("Ego motion not supported for env: {env}")

        # https://github.com/openai/gym/blob/master/gym/spaces/dict.py
        self.observation_space = spaces.Dict({'normal': spaces.Box(low=0, high=255,
                                                                   shape=(self.height, self.width, 1),
                                                                   dtype=np.uint8),
                                              'ego': spaces.Box(low=0, high=255,
                                                                shape=(self.ego_game.ego_h,
                                                                       self.ego_game.ego_w,
                                                                       1),
                                                                dtype=np.uint8)})

    def observation(self, frame):
        # Ego frame processing
        ego_frame = self.ego_game.find_character_in_frame(frame)
        ego_frame = cv2.cvtColor(ego_frame, cv2.COLOR_RGB2GRAY)
        ego_frame = cv2.resize(ego_frame, (self.ego_game.ego_w, self.ego_game.ego_h),
                               interpolation=cv2.INTER_AREA)

        # Previous 84x84 frame processing
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        res = {'normal': frame[:, :, None],
               'ego': ego_frame[:, :, None]}
        return res

class WarpEgo(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        # check that env is montezuma not something else
        gym.ObservationWrapper.__init__(self, env)
        # self.width = 84
        # self.height = 84
        self.width = 51
        self.height = 30

        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

        self.lower_color = np.array([199, 71, 71], dtype="uint8")
        self.upper_color = np.array([201, 73, 73], dtype="uint8")

    def find_character_in_frame(self, frame):
        mask = cv2.inRange(frame, self.lower_color, self.upper_color)
        output = cv2.bitwise_and(frame, frame, mask=mask)

        pix_x, pix_y, _ = np.where(output > 0)
        if pix_x.size != 0:
            pix_x = pix_x[np.where(pix_x > 19)]
            pix_y = pix_y[-pix_x.size:]

            # If array is even then median doesn't exist in the array, because it's the average
            # between the middle twos
            median_x = int(np.median(pix_x))
            while median_x not in pix_x:
                median_x += 1

            median_y = int(pix_y[np.where(pix_x == median_x)[0][0]])
        else:
            median_x = output.shape[0] // 2
            median_y = output.shape[1] // 2

        low_x = median_x-self.height
        high_x = median_x+self.height
        low_y = median_y-self.width
        high_y = median_y+self.width

        low_x = low_x if low_x > 0 else 0
        high_x = high_x if high_x < frame.shape[0] else frame.shape[0]
        low_y = low_y if low_y > 0 else 0
        high_y = high_y if high_y < frame.shape[1] else frame.shape[1]

        roi = frame[low_x:high_x, low_y:high_y]
        return roi

    def observation(self, frame):
        frame = self.find_character_in_frame(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.
        A single frame when using WarpFrame is 84x84x1
        So if we stack 4 frames then the shape is 84x84x4

        See Also
        --------
        rl_common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class DummyMontezumaInfoWrapper(gym.Wrapper):

    def __init__(self, env):
        super(DummyMontezumaInfoWrapper, self).__init__(env)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(pos_count=0,
                                   visited_rooms=set([0]))
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()

class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if d:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode']['rng_at_episode_start'] = self.rng_at_episode_start
        return ob, r, d, info

    def reset(self, **kwargs):
        self.rng_at_episode_start = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)


def make_atari(env_id, max_episode_steps=4500):
    env = gym.make(env_id)
    env._max_episode_steps = max_episode_steps*4
    assert 'NoFrameskip' in env.spec.id
    env = StickyActionEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if "Montezuma" in env_id or "Pitfall" in env_id:
        env = MontezumaInfoWrapper(env, room_address=3 if "Montezuma" in env_id else 1)
    else:
        env = DummyMontezumaInfoWrapper(env)
    env = AddRandomStateToInfo(env)
    return env


def wrap_deepmind(env, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if os.environ.get('EXPERIMENT_LVL') == 'ego':
        env = WarpFrame(env)
    else:
        env = OldWarpFrame(env)

    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    # env = NormalizeObservation(env)
    return env


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
