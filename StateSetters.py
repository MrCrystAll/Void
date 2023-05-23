import math
import random

import numpy as np
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, \
    CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED, BLUE_TEAM, ORANGE_TEAM
from rlgym_sim.utils.math import rand_vec3
from rlgym_sim.utils.state_setters import StateSetter
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi

DEG_TO_RAD = np.pi / 180


class AerialStateSetter(StateSetter):
    MID_AIR = 900
    LOW_AIR = 150
    HIGH_AIR = 1500
    CEILING = CEILING_Z


def compare_arrays_inferior(arr1: np.array, arr2: np.array) -> bool:
    return all(arr1[0:2] < arr2[0:2])


def _check_positions(state: StateWrapper) -> bool:
    CAR_DIM = np.array((50, 50, 50))

    # Checking all others for each car
    for car in state.cars:

        current_car = car

        for other in state.cars:
            if other == current_car:
                continue

            if compare_arrays_inferior(abs(current_car.position - other.position), CAR_DIM):
                print("Car too close, resetting")
                return False

        if compare_arrays_inferior(abs(current_car.position - state.ball.position), CAR_DIM):
            return False

    return True


class CustomStateSetter(StateSetter):
    """def reset(self, state_wrapper: StateWrapper):
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining
        for car in state_wrapper.cars:

            desired_car_pos = [random.randint(-4000, 4000), random.randint(-5000, 5000), random.randint(50, int(CEILING_Z) - int(BALL_RADIUS) * 2)]
            desired_yaw = np.pi / 2

            #print(desired_car_pos)

            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = desired_yaw

            elif car.team_num == ORANGE_TEAM:  # invert
                pos = [-1 * coord for coord in desired_car_pos]
                yaw = -1 * desired_yaw

            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = random.randint(0, 100)

        state_wrapper.ball.set_pos(random.randint(-3000, 3000), random.randint(-3000, 3000),
                                   random.randint(0 + int(BALL_RADIUS) * 2, int(CEILING_Z) - int(BALL_RADIUS) * 2))"""

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)


class ShotState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, 3000),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) + 1000, car.position.item(1) + 100),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class JumpShotState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, 2500),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=0,
                    yaw=90,
                    roll=0
                )

                ang_vel = (0, 0, 0)  # rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) + 1500, car.position.item(1) + 500),
                    z=CEILING_Z / 2
                )

                ball_speed = np.random.uniform(100, BALL_MAX_SPEED / 2)
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 2))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = (0, 0, 0)
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class SaveState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            if car.team_num == ORANGE_TEAM:
                car.set_pos(
                    random.uniform(-4096, 4096),
                    random.uniform(0, -3000),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)

                state_wrapper.ball.set_pos(
                    x=np.random.uniform(max(car.position.item(0) - 1000, -LIM_X),
                                        min(car.position.item(0) + 1000, LIM_X)),
                    y=np.random.uniform(car.position.item(1) - 1000, car.position.item(1) - 100),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z / 2),
                )

                ball_speed = np.random.exponential(-(BALL_MAX_SPEED / 3) / np.log(1 - 0.999))
                vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED / 3))
                state_wrapper.ball.set_lin_vel(*vel)

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
                state_wrapper.ball.set_ang_vel(*ang_vel)

            if car.team_num == BLUE_TEAM:
                car.set_pos(
                    random.randint(-2900, 2900),
                    -random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class AirDribble2Touch(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):

        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:

                car_x = random.randint(-3500, 3500)
                car_y = random.randint(-2000, 4000)
                car_z = np.random.uniform(500, LIM_Z)

                car.set_pos(
                    car_x,
                    car_y,
                    car_z
                )

                car_pitch_rot = random.uniform(-1, 1) * math.pi
                car_yaw_rot = random.uniform(-1, 1) * math.pi
                car_roll_rot = random.uniform(-1, 1) * math.pi

                car.set_rot(
                    car_pitch_rot,
                    car_yaw_rot,
                    car_roll_rot
                )

                car_lin_y = np.random.uniform(300, CAR_MAX_SPEED)

                car.set_lin_vel(
                    0 + random.uniform(-150, 150),
                    car_lin_y,
                    0 + random.uniform(-150, 150)
                )

                state_wrapper.ball.set_pos(
                    car_x + random.uniform(-150, 150),
                    car_y + random.uniform(0, 150),
                    car_z + random.uniform(0, 150)
                )

                ball_lin_y = car_lin_y + random.uniform(-150, 150)

                state_wrapper.ball.set_lin_vel(
                    0 + random.uniform(-150, 150),
                    ball_lin_y,
                    0 + random.uniform(-150, 150)
                )
                car.boost = np.random.uniform(0.4, 1)

            else:
                car.set_pos(
                    random.randint(-2900, 2900),
                    random.randint(3000, 5120),
                    17
                )

                vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
                car.set_ang_vel(*ang_vel)
                car.boost = np.random.uniform(0, 1)


class DefaultState(StateSetter):
    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        rand_default_or_diff = random.randint(0, 3)
        # rand_default_or_diff = 0
        # DEFAULT KICKOFFS POSITIONS
        if rand_default_or_diff == 0:

            # possible kickoff indices are shuffled
            spawn_inds = [0, 1, 2, 3, 4]
            random.shuffle(spawn_inds)

            blue_count = 0
            orange_count = 0
            for car in state_wrapper.cars:
                pos = [0, 0, 0]
                yaw = 0
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_BLUE_POS[spawn_inds[blue_count]]
                    yaw = self.SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                    blue_count += 1
                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    pos = self.SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                    yaw = self.SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                    orange_count += 1
                # set car state values
                car.set_pos(*pos)
                car.set_rot(yaw=yaw)
                car.boost = 0.33
        else:
            """SPAWN POS RANGE
            Y = CENTER BACK KICKOFF [-4608, 0], [4608, 0]; BACK LEFT/RIGHT KICKOFFS [-3000, 0], [3000,0]"""

            # print("Advanced Kickoffs")
            # SPAWN POSITION
            spawn_inds = [0, 1, 2, 3, 4]
            spawn_pos = np.random.choice(spawn_inds)

            same_pos = random.randint(0, 1)
            # print(f"Same position = {same_pos}")

            # RIGHT CORNER
            if spawn_pos == 0:
                # print("Right Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / -2048
                    posX = np.random.uniform(-2048, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 200)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / -2048
                        blue_posX = np.random.uniform(-2048, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / 2048
                        orange_posX = np.random.uniform(2048, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            # LEFT CORNER
            elif spawn_pos == 1:
                # print("Left Corner")
                # SAME POS
                if same_pos == 1:
                    slope = -2560 / 2048
                    posX = np.random.uniform(2048, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -2560 / 2048
                        blue_posX = np.random.uniform(2048, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        slope = 2560 / -2048
                        orange_posX = np.random.uniform(-2048, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 2:
                # print("Back Right")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / -256
                    posX = np.random.uniform(-256, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (0, posY, posZ)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / -256
                        blue_posX = np.random.uniform(-256, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / 256
                        orange_posX = np.random.uniform(256, 0)
                        orange_posY = slope * orange_posX
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 3:
                # print("Back Left")
                # SAME POS
                if same_pos == 1:
                    slope = -3840 / 256
                    posX = np.random.uniform(256, 0)
                    posY = slope * posX
                    posZ = np.random.uniform(17, 500)

                    if posY < -3000:
                        blue_spawn_pos = (posX, posY, posZ)
                        orange_spawn_pos = (-abs(posX), abs(posY), posZ)
                    else:
                        blue_spawn_pos = (posX, 0, 17)
                        orange_spawn_pos = (0, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        slope = -3840 / 256
                        blue_posX = np.random.uniform(256, 0)
                        blue_posY = slope * blue_posX
                        posZ = np.random.uniform(17, 500)
                        if blue_posY < -3000:
                            blue_spawn_pos = (blue_posX, blue_posY, posZ)
                        else:
                            blue_spawn_pos = (0, blue_posY, posZ)

                        slope = 3840 / -256
                        orange_posX = np.random.uniform(-256, 0)
                        orange_posY = slope * orange_posX
                        posZ = np.random.uniform(17, 500)
                        if orange_posX > 3000:
                            orange_spawn_pos = (orange_posX, orange_posY, posZ)
                        else:
                            orange_spawn_pos = (0, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        #  print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
            elif spawn_pos == 4:
                # print("Far Back Center")
                # SAME POS
                if same_pos == 1:
                    posX = 0
                    posY = np.random.uniform(-4608, 0)
                    posZ = np.random.uniform(17, 500)
                    blue_spawn_pos = (posX, posY, posZ)
                    orange_spawn_pos = (posX, abs(posY), posZ)
                    # print(f"Blue Spawn: {blue_spawn_pos} | Orange Spawn {orange_spawn_pos}")

                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0
                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                        # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)
                else:
                    for car in state_wrapper.cars:
                        final_pos = [0, 0, 0]
                        final_yaw = 0

                        blue_posX = 0
                        blue_posY = np.random.uniform(-4608, 0)
                        posZ = np.random.uniform(17, 500)
                        blue_spawn_pos = (blue_posX, blue_posY, posZ)

                        orange_posX = 0
                        orange_posY = np.random.uniform(4608, 0)
                        posZ = np.random.uniform(17, 500)
                        orange_spawn_pos = (orange_posX, orange_posY, posZ)

                        if car.team_num == BLUE_TEAM:
                            final_pos = blue_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Blue Final Spawn: {final_pos} | Yaw: {final_yaw}")
                        elif car.team_num == ORANGE_TEAM:
                            final_pos = orange_spawn_pos
                            final_yaw = random.uniform(-1, 1)
                            # print(f"Orange Final Spawn: {final_pos} | Yaw: {final_yaw}")

                        car.set_pos(*final_pos)
                        car.set_rot(final_yaw)
                        car.boost = np.random.uniform(0, 0.4)
                        vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
                        car.set_lin_vel(*vel)


class AirDribbleSetup(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        axis_inverter = 1 if random.randrange(2) == 1 else -1
        team_side = 0 if random.randrange(2) == 1 else 1
        team_inverter = 1 if team_side == 0 else -1

        # if only 1 play, team is always 0

        ball_x_pos = 3000 * axis_inverter
        ball_y_pos = random.randrange(7600) - 3800
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + (random.randrange(1000) - 500)) * axis_inverter
        ball_y_vel = random.randrange(1000) * team_inverter
        ball_z_vel = 0
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        chosen_car = [car for car in state_wrapper.cars if car.team_num == team_side][0]
        # if randomly pick, chosen_car is from orange instead

        car_x_pos = 2500 * axis_inverter
        car_y_pos = ball_y_pos
        car_z_pos = 27

        yaw = 0 if axis_inverter == 1 else 180
        car_pitch_rot = 0 * DEG_TO_RAD
        car_yaw_rot = (yaw + (random.randrange(40) - 20)) * DEG_TO_RAD
        car_roll_rot = 0 * DEG_TO_RAD

        chosen_car.set_pos(car_x_pos, car_y_pos, car_z_pos)
        chosen_car.set_rot(car_pitch_rot, car_yaw_rot, car_roll_rot)
        chosen_car.boost = 100

        for car in state_wrapper.cars:
            if car is chosen_car:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)


class NaNState(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=1943.54,
            y=-776.12,
            z=406.78,
        )
        vel = (122.79422, -234.43246, 259.7227)
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = (-0.11868675, 0.10540164, -0.05797875)
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:

            if car.team_num == 0:
                car.set_pos(
                    x=1943.54,
                    y=-776.12,
                    z=406.78,
                )

                vel = (-303.3699951171875, -214.94998168945312, -410.7699890136719)
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = (-0.91367, -1.7385, 0.28524998)
                car.set_ang_vel(*ang_vel)
                car.boost = 0.593655776977539
            else:
                car.set_pos(
                    x=1917.4401,
                    y=-738.27997,
                    z=462.01,
                )

                vel = (142.98796579, 83.07732574, -20.63784965)
                car.set_lin_vel(*vel)

                car.set_rot(
                    pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                    yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                    roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
                )

                ang_vel = (-0.22647299, 0.10713767, 0.24121591)
                car.set_ang_vel(*ang_vel)
                car.boost = 0.8792996978759766


class SideHighRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        sidepick = random.randrange(2)

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        ball_x_pos = 3000 * side_inverter
        ball_y_pos = random.randrange(1500) - 750
        ball_z_pos = BALL_RADIUS
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (2000 + random.randrange(1000) - 500) * side_inverter
        ball_y_vel = random.randrange(1500) - 750
        ball_z_vel = random.randrange(300)
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car_blue = [car for car in state_wrapper.cars if car.team_num == 0][0]

        # blue car setup
        blue_pitch_rot = 0 * DEG_TO_RAD
        blue_yaw_rot = 90 * DEG_TO_RAD
        blue_roll_rot = 90 * side_inverter * DEG_TO_RAD
        wall_car_blue.set_rot(blue_pitch_rot, blue_yaw_rot, blue_roll_rot)

        blue_x = 4096 * side_inverter
        blue_y = -2500 + (random.randrange(500) - 250)
        blue_z = 600 + (random.randrange(400) - 200)
        wall_car_blue.set_pos(blue_x, blue_y, blue_z)
        wall_car_blue.boost = 100

        # orange car setup
        wall_car_orange = None
        if len(state_wrapper.cars) > 1:
            wall_car_orange = [car for car in state_wrapper.cars if car.team_num == 1][0]
            # orange car setup
            orange_pitch_rot = 0 * DEG_TO_RAD
            orange_yaw_rot = -90 * DEG_TO_RAD
            orange_roll_rot = -90 * side_inverter * DEG_TO_RAD
            wall_car_orange.set_rot(orange_pitch_rot, orange_yaw_rot, orange_roll_rot)

            orange_x = 4096 * side_inverter
            orange_y = 2500 + (random.randrange(500) - 250)
            orange_z = 400 + (random.randrange(400) - 200)
            wall_car_orange.set_pos(orange_x, orange_y, orange_z)
            wall_car_orange.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car_orange or car is wall_car_blue:
                continue

            # set all other cars randomly in the field
            car.set_pos(random.randrange(2944) - 1472, random.randrange(3968) - 1984, 0)
            car.set_rot(0, (random.randrange(360) - 180) * (3.1415927 / 180), 0)


class ShortGoalRoll(StateSetter):

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        if len(state_wrapper.cars) > 1:
            defense_team = random.randrange(2)
        else:
            defense_team = 0
        sidepick = random.randrange(2)

        defense_inverter = 1
        if defense_team == 0:
            # change side
            defense_inverter = -1

        side_inverter = 1
        if sidepick == 1:
            # change side
            side_inverter = -1

        # MAGIC NUMBERS ARE FROM MANUAL CALIBRATION AND WHAT FEELS RIGHT

        x_random = random.randrange(446)
        ball_x_pos = (-2850 + x_random) * side_inverter
        ball_y_pos = (5120 - BALL_RADIUS) * defense_inverter
        ball_z_pos = 1400 + random.randrange(400) - 200
        state_wrapper.ball.set_pos(ball_x_pos, ball_y_pos, ball_z_pos)

        ball_x_vel = (1000 + random.randrange(400) - 200) * side_inverter
        ball_y_vel = 0
        ball_z_vel = 550
        state_wrapper.ball.set_lin_vel(ball_x_vel, ball_y_vel, ball_z_vel)

        wall_car = [car for car in state_wrapper.cars if car.team_num == defense_team][0]

        wall_car_x = (2000 - random.randrange(500)) * side_inverter
        wall_car_y = 5120 * defense_inverter
        wall_car_z = 1000 + (random.randrange(500) - 500)
        wall_car.set_pos(wall_car_x, wall_car_y, wall_car_z)

        wall_pitch_rot = (0 if side_inverter == -1 else 180) * DEG_TO_RAD
        wall_yaw_rot = 0 * DEG_TO_RAD
        wall_roll_rot = -90 * defense_inverter * DEG_TO_RAD
        wall_car.set_rot(wall_pitch_rot, wall_yaw_rot, wall_roll_rot)
        wall_car.boost = 25

        if len(state_wrapper.cars) > 1:
            challenge_car = [car for car in state_wrapper.cars if car.team_num != defense_team][0]
            challenge_car.set_pos(0, 1000 * defense_inverter, 0)

            challenge_pitch_rot = 0 * DEG_TO_RAD
            challenge_yaw_rot = 90 * defense_inverter * DEG_TO_RAD
            challenge_roll_rot = 0 * DEG_TO_RAD
            challenge_car.set_rot(challenge_pitch_rot, challenge_yaw_rot, challenge_roll_rot)
            challenge_car.boost = 100

        for car in state_wrapper.cars:
            if len(state_wrapper.cars) == 1 or car is wall_car or car is challenge_car:
                continue

            car.set_pos(random.randrange(2944) - 1472, (-4500 + random.randrange(500) - 250) * defense_inverter, 0)
            car.set_rot(0, (random.randrange(360) - 180) * DEG_TO_RAD, 0)


class StandingBallState(StateSetter):

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(x=random.Random().randint(-SIDE_WALL_X + 500, SIDE_WALL_X - 500),
                                   y=0,
                                   z=random.Random().randint(1000, 1300))

        state_wrapper.ball.set_lin_vel(x=random.uniform(-1000, 1000),
                                       y=random.uniform(-1000, 1000),
                                       z=random.uniform(400, 800)
                                       )

        blue_team = []
        orange_team = []

        for player in state_wrapper.cars:

            x = 0
            y = 0
            yaw = 0
            if player.team_num == ORANGE_TEAM:

                y = random.uniform(1000, 2000)
                yaw = -0.5 * np.pi

                if len(orange_team) >= 1:
                    choice = random.choice(orange_team)

                    if choice.position.item(0) > 3000:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) - 200)

                    elif choice.position.item(0) < -3000:
                        x = random.Random().randint(choice.position.item(0) + 200,
                                                    choice.position.item(0) + 500)
                    else:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) + 500)

                else:
                    x = random.Random().randint(state_wrapper.ball.position.item(0) - 300,
                                                state_wrapper.ball.position.item(0) + 300)

                orange_team.append(player)
            else:
                y = random.uniform(-2000, -1000)
                yaw = 0.5 * np.pi

                if len(blue_team) >= 1:
                    choice = random.choice(blue_team)

                    if choice.position.item(0) > 3000:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) - 200)

                    elif choice.position.item(0) < -3000:
                        x = random.Random().randint(choice.position.item(0) + 200,
                                                    choice.position.item(0) + 500)
                    else:
                        x = random.Random().randint(choice.position.item(0) - 500,
                                                    choice.position.item(0) + 500)

                else:
                    x = random.Random().randint(
                        state_wrapper.ball.position.item(0) - 300,
                        state_wrapper.ball.position.item(0) + 300)

            player.set_pos(
                x=x,
                y=y,
                z=30)
            player.set_rot(yaw=yaw)
            player.boost = 100


class AerialBallState(StateSetter):
    def __init__(self):
        super().__init__()
        self.observators = []

    def reset(self, state_wrapper: StateWrapper):

        if len(self.observators) != 0:
            distances = []
            for player in state_wrapper.cars:
                dist = math.sqrt((state_wrapper.ball.position.item(0) - player.position.item(0)) ** 2 +
                                 (state_wrapper.ball.position.item(1) - player.position.item(1)) ** 2 +
                                 (state_wrapper.ball.position.item(2) - player.position.item(2)) ** 2)
                distances.append(dist)

            distance = np.mean(distances)

            for obs in self.observators:
                obs.update(distance=float(distance))

        xthreshold = 1800
        ythreshold = 1200
        zthreshold = 500

        # Ball position
        state_wrapper.ball.set_pos(x=np.random.randint(-SIDE_WALL_X + xthreshold,
                                                       SIDE_WALL_X - xthreshold),

                                   y=np.random.randint(-BACK_WALL_Y + ythreshold,
                                                       BACK_WALL_Y - ythreshold),

                                   z=np.random.randint(zthreshold,
                                                       CEILING_Z - 2 * zthreshold)
                                   )

        # Ball speed
        state_wrapper.ball.set_lin_vel(x=np.random.randint(500, 1000),
                                       y=np.random.randint(500, 1000),
                                       z=np.random.randint(500, 1000)
                                       )

        for player in state_wrapper.cars:
            player.set_pos(x=np.random.randint(-SIDE_WALL_X + xthreshold,
                                               SIDE_WALL_X - xthreshold),
                           y=np.random.randint(-BACK_WALL_Y + ythreshold,
                                               BACK_WALL_Y - ythreshold),
                           z=30)

            player.boost = 100

            # player.set_rot(pitch=random.Random().random() * np.pi,
            #                yaw=random.Random().random() * np.pi,
            #                roll=random.Random().random() * np.pi
            #                )
            #
            # player.set_lin_vel(x=np.random.randint(0, constants.CAR_MAX_SPEED),
            #                    y=np.random.randint(0, constants.CAR_MAX_SPEED),
            #                    z=np.random.randint(0, constants.CAR_MAX_SPEED)
            #                    )

    '''def bind_to(self, ball_register: BallDistRegisterer):
        self.observators.append(ball_register)'''


def probs_function(number):
    if number < 0.5:
        return 0
    else:
        return -4 * pow(number + 0.5, 2) + 1


class DynamicScoredReplaySetter(ReplaySetter):
    def __init__(self, ones_file: str, twos_file: str, threes_file: str):
        print("Loading 1s replays...")
        data1s = np.load(ones_file)
        print("Loading 2s replays...")
        data2s = np.load(twos_file)
        print("Loading 3s replays...")
        data3s = np.load(threes_file)
        print("Loading done")

        mask = ~np.isnan(data1s["states"]).any(axis=1) & ~np.isnan(data1s["scores"])
        self.states1s = data1s["states"][mask]
        self.scores1s = data1s["scores"][mask]
        self.scores = self.scores1s
        self.probs1s = self.generate_probabilities()

        mask = ~np.isnan(data2s["states"]).any(axis=1) & ~np.isnan(data2s["scores"])
        self.states2s = data2s["states"][mask]
        self.scores2s = data2s["scores"][mask]
        self.scores = self.scores2s
        self.probs2s = self.generate_probabilities()

        mask = ~np.isnan(data3s["states"]).any(axis=1) & ~np.isnan(data3s["scores"])
        self.states3s = data3s["states"][mask]
        self.scores3s = data3s["scores"][mask]
        self.scores = self.scores3s
        self.probs3s = self.generate_probabilities()

        self.scores = self.scores1s
        super().__init__(self.states1s)


    def generate_probabilities(self):
        scores = np.vectorize(probs_function)(self.scores)
        probs = scores / scores.sum()
        return probs

    def reset(self, state_wrapper: StateWrapper):
        if len(state_wrapper.cars) == 6:
            self.states = self.states3s
            self.scores = self.scores3s
            self.probabilities = self.probs3s
        elif len(state_wrapper.cars) == 4:
            self.states = self.states2s
            self.scores = self.scores2s
            self.probabilities = self.probs2s
        else:
            self.states = self.states1s
            self.scores = self.scores1s
            self.probabilities = self.probs1s

        self.generate_probabilities()
        super(DynamicScoredReplaySetter, self).reset(state_wrapper)


all_states = [DefaultState()]
states = [
    CustomStateSetter(),
    DefaultState(),
    ShotState(),
    SaveState(),
    AerialBallState(),
    DynamicScoredReplaySetter(
        "replays/states_scores_duels.npz",
        "replays/states_scores_doubles.npz",
        "replays/states_scores_standard.npz"
    )
]


class ProbabilisticStateSetter(StateSetter):

    def __init__(self, verbose: int = 1, training: bool = True):
        self.old_state = None
        self.verbose = verbose
        self.training = training

    def reset(self, state_wrapper: StateWrapper):
        if self.verbose >= 1:
            if self.old_state:
                print(f"State {self.old_state.__class__.__name__} finished, choosing a new state...")
            else:
                print(f"Training started, choosing a new state...")

        if self.training:
            random_location_prob = 1
            kickoff_prob = 1
            shot_prob = 49
            save_prob = 49
            aerial_prob = 20
            scored_replay_prob = 20

            all_probs = [random_location_prob, kickoff_prob, shot_prob, save_prob, aerial_prob, scored_replay_prob]

            selected_state = random.choices(states, weights=all_probs, k=1)[0]
        else:
            selected_state = all_states.pop(0)

        self.old_state = selected_state

        if self.verbose >= 1:
            print(f"New state is {selected_state.__class__.__name__}")

        # while not _check_positions(state_wrapper):
        selected_state.reset(state_wrapper)
