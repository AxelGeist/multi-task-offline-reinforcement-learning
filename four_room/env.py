from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from itertools import product
import random
from typing import Any

class FourRoomsEnv(MiniGridEnv):

    """
    ### Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.

    ### Mission Space

    "reach the goal"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of 1 - 0.9 * (step_count / max_steps) is given for success, and 0 for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    """
    @staticmethod
    def valid_positions(size):
        """
            generate a list of allowed agent positions and goal positions
            to make sampling from it easier
        """
        valid_agent_pos = list()
        valid_goal_pos = list()

        # loop through the 4 rooms:
        for i in range(0,2):
            for j in range(0,2):
                # every room has (size // 2 - 1)**2 possible locations
                for k in range(0,size // 2 - 1):
                    for l in range(0,size // 2 - 1):
                        pos = (i*(size // 2) + 1 + k, j*(size // 2) + 1 + l)
                        valid_agent_pos.append(pos)
                        valid_goal_pos.append(pos)

        valid_doors_pos = list(product(list(range(0, size//2 - 1)), repeat=4))
        
        return valid_agent_pos, valid_goal_pos, valid_doors_pos


    def __init__(self, agent_pos=None, agent_dir=None, goal_pos=None, doors_pos=None, max_steps=100, **kwargs):
        """
            This code is only works for environment size 9

            Parameters
            ----------
            agent_pos : list
                List of 2-tuples containing agent starting locations
            agent_dir : list
                List of integers containing agent directions (in range [0,3])
            goal_pos : list
                List of 2-tuples containing goal locations
            doors_pos : list
                List of 4-tuples containing door locations
                Locations are integers in the range [0,2] and in order of wall segments: (up, down, left right)
            max_steps : int
                Number of steps after which to timeout the agent
        """
        self._agent_pos_list = agent_pos
        self._goal_pos_list = goal_pos
        self._doors_pos_list = doors_pos
        self._agent_dir_list = agent_dir

        # an index into agent_pos, goal_pos and doors_pos with which we will initialise the environment
        self._list_idx = 0

        if (agent_pos is not None) and (agent_dir is not None) and (goal_pos is not None) and (doors_pos is not None):
            self._list_size = len(self._agent_pos_list)
            assert len(self._agent_dir_list) == self._list_size
            assert len(self._goal_pos_list) == self._list_size
            assert len(self._doors_pos_list) == self._list_size


        self.size = 9
        mission_space = MissionSpace(mission_func=lambda: "reach the goal")

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            highlight=False,
            **kwargs
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,):
        if seed and self._agent_pos_list is not None:
            random.seed(seed)
            self._list_idx = random.randint(0, len(self._agent_pos_list)-1)

            # shuffle the order of the configurations
            # this ensures a different seed will encounter the training configurations in a different order
            initial_configurations = list(zip(self._agent_pos_list, self._agent_dir_list, self._goal_pos_list, self._doors_pos_list))
            random.shuffle(initial_configurations)
            self._agent_pos_list, self._agent_dir_list, self._goal_pos_list, self._doors_pos_list = zip(*[i for i in initial_configurations])

        return super().reset(seed=seed, options=options)


    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        if self._doors_pos_list is not None:
            doors_pos = self._doors_pos_list[self._list_idx]
            agent_dir = self._agent_dir_list[self._list_idx]
        else:
            doors_pos = (self._rand_int(0, 3), self._rand_int(0, 3), self._rand_int(0, 3), self._rand_int(0, 3))
            agent_dir = self._rand_int(0, 4)


        # For each row of rooms
        for j in range(0, 2):
            # For each column or rooms
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, yT + 1 + doors_pos[j])
                    self.grid.set(*pos, None)

                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (xL + 1 + doors_pos[2 + i], yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_pos_list is not None:
            self.agent_pos = self._agent_pos_list[self._list_idx]
            self.grid.set(*self.agent_pos, None)
            self.agent_dir = agent_dir
        else:
            self.place_agent()

        if self._goal_pos_list is not None:
            goal = Goal()
            self.goal_pos = self._goal_pos_list[self._list_idx]
            self.put_obj(goal, *self.goal_pos)
            goal.init_pos, goal.cur_pos = self.goal_pos
        else:
            self.place_obj(Goal())

        if self._agent_pos_list is not None:
            # assumes _gen_grid() is only called once when reset() is called
            self._list_idx = (self._list_idx + 1) % self._list_size


class FourRoomsNoRotateEnv(FourRoomsEnv):

    """
    ### Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. 
    The agent is randomly placed in any of the four rooms and the goal is always 
    in the lower right room.

    ### Mission Space

    "reach the goal"

    ### Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ### Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [gym_minigrid/minigrid.py](gym_minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ### Rewards

    A reward of 1 - 0.9 * (step_count / max_steps) is given for success, and 0 for failure.

    ### Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ### Registered Configurations

    - `MiniGrid-FourRooms-v0`

    """
    @staticmethod
    def valid_positions(size):
        # generate a list of allowed agent positions and goal positions
        # to make sampling from it easier
        valid_agent_pos = list()
        valid_goal_pos = list()

        # goal can only be in the lower right room
        # every room has (size // 2 - 1)**2 possible locations
        for k in range(0,size // 2 - 1):
            for l in range(0,size // 2 - 1):
                pos = ((size // 2) + 1 + k, (size // 2) + 1 + l)
                valid_goal_pos.append(pos)

        # agent can be in any of the 4 rooms
        for i in range(0,2):
            for j in range(0,2):
                # every room has (size // 2 - 1)**2 possible locations
                for k in range(0,size // 2 - 1):
                    for l in range(0,size // 2 - 1):
                        pos = (i*(size // 2) + 1 + k, j*(size // 2) + 1 + l)
                        valid_agent_pos.append(pos)

        valid_doors_pos = list(product(list(range(0, size//2 - 1)), repeat=4))
        
        return valid_agent_pos, valid_goal_pos, valid_doors_pos
