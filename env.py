

import numpy as np

from utils import quaternion_multiply, quaternion_inverse, log_quaternion

norm = np.linalg.norm


class MobileRobot:
    """
    Simulate a sequence as a mobile robot.
    """
    def __init__(self, times, time2rgb, dim_control, time2pos=None, time2vopos=None,
                 no_model=False, norm_xyz=1., norm_q=1.):
        self.timestamps = times
        self.time2rgb = time2rgb
        self.dim_control = dim_control
        self.time2pos = time2pos
        self.time2vopos = time2vopos if time2vopos is not None else self.time2pos
        self.no_model = no_model
        self.norm_xyz = norm_xyz
        self.norm_q = norm_q

        self.pos = None
        self.time = None
        self.time_idx = 0
        self.reset()

    def render(self, step=None):
        if step is None:
            step = self.time_idx
        time = self.timestamps[step]
        return self.time2rgb[time]

    def reset(self, time_idx=0):
        self.time_idx = time_idx
        self.time = self.timestamps[time_idx]
        if self.time2pos is not None:
            self.pos = self.time2pos[self.time]
        return self.render()

    def get_pos(self):
        if self.time2pos is None:
            raise Exception("The ground truth position is not available!")
        self.pos = self.time2pos[self.time]
        xyz = self.pos[:3] / self.norm_xyz
        q = self.pos[3:]
        log_q = log_quaternion(q) / self.norm_q
        return np.concatenate((xyz, log_q)).reshape(-1, 1)

    def step(self):
        """
        Return the next step image, the relative action (vel + quaternion), and time diff.

        If self.no_model is not set, the real velocity & real relative rotation (as a quaternion) will be returned.

        Otherwise, a vector of action = [0, 0, 0, 1, 0, 0, 0] is returned, indicating no
        translational movements (velocity  = action[:3] = 0), and no orientation change, i.e., a constant transition.
        """
        new_time = self.timestamps[self.time_idx + 1]
        dt = new_time - self.time

        new_pos = self.time2pos[new_time]
        vel = (self.time2vopos[new_time][:3] - self.time2vopos[self.time][:3]) / dt
        qt = self.time2vopos[self.time][3:]
        qt_1 = self.time2vopos[new_time][3:]
        # for quaternion, time is not considered
        q = quaternion_multiply(qt_1, quaternion_inverse(qt))
        # combine
        action = np.concatenate((vel, q), axis=0)

        if self.no_model:
            action = np.zeros_like(action)
            action[3] = 1.  # quaternion constant model
        self.pos = new_pos
        self.time_idx += 1
        self.time = new_time
        return self.render(), action.reshape(-1, 1), dt

    @property
    def horizon(self):
        return len(self.timestamps)

    @property
    def dim_state(self):
        return len(self.get_pos())
