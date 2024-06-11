import numpy as np

def obs_to_img(obs):
    """
        Turn a numpy observation array into an image array that can be plotted with matplotlib
    """
    walls = obs[2]
    lower_right = np.array([np.where(walls.sum(axis=1) == 9)[0][0], np.where(walls.sum(axis=0) == 9)[0][0]])
    shift = np.array([8,8]) - lower_right
    if np.where(walls.sum(axis=1) == 9)[0][0] == 0 and np.where(walls.sum(axis=1) == 9)[0][1] == 8:
        shift[0] = 0
    if np.where(walls.sum(axis=0) == 9)[0][0] == 0 and np.where(walls.sum(axis=0) == 9)[0][1] == 8:
        shift[1] = 0

    uncentered_obs = np.roll(obs, tuple(shift), axis=(1,2))

    agent_color = np.tile(np.array([[[255,0,0]]]), (9,9,1))
    agent_dir_color = np.tile(np.array([[[255,128,128]]]), (9,9,1))
    goal_color = np.tile(np.array([[[0,255,0]]]), (9,9,1))
    wall_color = np.tile(np.array([[[128,128,128]]]), (9,9,1))

    agent_channel = np.repeat(np.expand_dims(uncentered_obs[0], axis=2), repeats=3, axis=2)
    dir_channel = np.repeat(np.expand_dims(uncentered_obs[1], axis=2), repeats=3, axis=2)
    goal_channel = np.repeat(np.expand_dims(uncentered_obs[3], axis=2), repeats=3, axis=2)
    wall_channel = np.repeat(np.expand_dims(uncentered_obs[2], axis=2), repeats=3, axis=2)
    img = agent_channel*agent_color  + dir_channel*agent_dir_color + goal_channel*goal_color + wall_channel*wall_color
    img = img.astype(np.uint8)

    return img

def obs_to_state(obs):
    """
        Turn a numpy observation array into a tuple of the form:
        (player location x, player location y, player direction, goal location x, goal location y, 
            door position up, door position down, door position left, door position right)
    """
    walls = obs[2]
    lower_right = np.array([np.where(walls.sum(axis=1) == 9)[0][0], np.where(walls.sum(axis=0) == 9)[0][0]])
    shift = np.array([8,8]) - lower_right
    if np.where(walls.sum(axis=1) == 9)[0][0] == 0 and np.where(walls.sum(axis=1) == 9)[0][1] == 8:
        shift[0] = 0
    if np.where(walls.sum(axis=0) == 9)[0][0] == 0 and np.where(walls.sum(axis=0) == 9)[0][1] == 8:
        shift[1] = 0

    uncentered_obs = np.roll(obs, tuple(shift), axis=(1,2))
    player_loc = (np.where(uncentered_obs[0] == 1)[1][0], np.where(uncentered_obs[0] == 1)[0][0])
    player_dir_loc = (np.where(uncentered_obs[1] == 1)[1][0], np.where(uncentered_obs[1] == 1)[0][0])
    player_dir_loc = np.array(player_loc) - np.array(player_dir_loc)
    if player_dir_loc[0] == 1 and  player_dir_loc[1] == 0:
        # left
        player_dir = 2
    if player_dir_loc[0] == 0 and  player_dir_loc[1] == -1:
        # down
        player_dir = 1
    if player_dir_loc[0] == -1 and  player_dir_loc[1] == 0:
        # right
        player_dir = 0
    if player_dir_loc[0] == 0 and  player_dir_loc[1] == 1:
        # up
        player_dir = 3
    
    goal_loc = (np.where(uncentered_obs[3] == 1)[1][0], np.where(uncentered_obs[3] == 1)[0][0])

    walls = uncentered_obs[2]
    doors_pos = (*(np.where(walls[:, 4] == 0)[0] - np.array([1, 5])), *(np.where(walls[4, :] == 0)[0] - np.array([1, 5])))

    return (*player_loc, player_dir, *goal_loc, *doors_pos)