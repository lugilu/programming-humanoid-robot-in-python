'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello, leftBackToStand, leftBellyToStand
import numpy as np


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.start_time = None # Starting time of interpolation

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        # YOUR CODE HERE
        names, times, keys = keyframes
        if not self.start_time:
            self.start_time = perception.time

        current_time = perception.time - self.start_time

        for index, joint_name in enumerate(names):
            interpol_times = times[index]
            all_angles = keys[index]    # angle and handles are in there

            if joint_name not in perception.joint.keys():
                continue

            time_index = "Past_Time_Frame"
            for index, time_point in enumerate(interpol_times): # find in between which time point current time is
                if current_time < time_point:
                    time_index = index # This means current_time is between time[index-1] and time[index]
                    break

            # do nothing when over last time point
            if time_index == "Past_Time_Frame":
                continue

            # Now come the four points needed for the bezier curve. Each point p = (x coord, y coord)
            if time_index == 0:     # Before defined time point
                p0 = (0, perception.joint[joint_name])
                p1 = (- all_angles[0][1][1], all_angles[0][1][2])
                p2 = (interpol_times[0] + all_angles[0][1][1], all_angles[0][0] + all_angles[0][1][2])
                p3 = (interpol_times[0], all_angles[0][0])

            else:
                left_point_angles, right_point_angles = all_angles[time_index - 1], all_angles[time_index]
                p0 = (interpol_times[time_index - 1], left_point_angles[0])
                p1 = (interpol_times[time_index - 1] + left_point_angles[2][1], left_point_angles[0] + left_point_angles[2][2])
                p2 = (interpol_times[time_index] + right_point_angles[1][1], right_point_angles[0] + right_point_angles[1][2])
                p3 = (interpol_times[time_index], right_point_angles[0])

            target_joints[joint_name] = self.cubic_bezier_angle(p0, p1, p2, p3, current_time)

        return target_joints

    def cubic_bezier_angle(self, p0, p1, p2, p3, current_time):
        """
        This function applies the cubic bezier formula with 4 points
        @param:
        We have x_point = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + \
                    3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
        But here x_point is known, so we need to solve for t. This is done by factorising the factors of t
        After factorization you get the factors of t**0, t**1, t**2, t**3 --> Let numpy solve this
        There will be exaclty one none complex solution --> cf analysis
        We dont want polynomial = 0 but polynomial = current_time, so we subtract time from factor 0
        """
        factor_0 = np.round(p0[0] - current_time, decimals=4)
        factor_1 = np.round(-3 * p0[0] + 3 * p1[0], decimals=4)
        factor_2 = np.round(3 * p0[0] - 6 * p1[0] + 3 * p2[0], decimals=4)
        factor_3 = np.round(- 1 * p0[0] + 3 * p1[0] -3 * p2[0] + p3[0], decimals=4)
        solve_points = np.roots([factor_3, factor_2, factor_1, factor_0])

        t = 0
        for solved in solve_points:
            if solved.imag == 0:
                t = float(solved.real)
                break

        #print "These are: ", t, factor_3, factor_2, factor_1, factor_0

        y_point = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + \
                    3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]

        return y_point



if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    # hello() works, other ones nearly makes it but cant finish the last movement to fully get up. Mabye more/better pid tuning needed
    agent.run()
