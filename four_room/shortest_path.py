import networkx as nx
from four_room.env import FourRoomsEnv
import numpy as np


def create_maze_graph(topology):
    """
        We turn our grid world into a graph as follows.
        - Every node is a combination of agent location (x,y) and agent direction
        - We add directed edges between nodes if the agent can transition from one to the other
    """

    graph = nx.DiGraph()
    valid_agent_positions, _, _ = FourRoomsEnv.valid_positions(9)

    ### generate 4 rooms
    # add the nodes
    for dir in range(4):
        graph.add_nodes_from([(x,y,dir) for x,y in valid_agent_positions])

    # add the edges in the up direction
    for room_x, room_y in [(1,1), (1,5), (5,1), (5,5)]:
        for i in range(0, 3):
            for j in range(1,3):
                graph.add_edge((room_x + i, room_y + j, 3), (room_x + i, room_y + j - 1, 3))
    # add the edges in the right direction
    for room_x, room_y in [(1,1), (1,5), (5,1), (5,5)]:
        for i in range(0, 2):
            for j in range(0,3):
                graph.add_edge((room_x + i, room_y + j, 0), (room_x + i + 1, room_y + j, 0))
    # add the edges in the down direction
    for room_x, room_y in [(1,1), (1,5), (5,1), (5,5)]:
        for i in range(0, 3):
            for j in range(0,2):
                graph.add_edge((room_x + i, room_y + j, 1), (room_x + i, room_y + j + 1, 1))
    # add the edges in the left direction
    for room_x, room_y in [(1,1), (1,5), (5,1), (5,5)]:
        for i in range(1, 3):
            for j in range(0,3):
                graph.add_edge((room_x + i, room_y + j, 2), (room_x + i - 1, room_y + j, 2))

    ## connect the direction layers together
    for room_x, room_y in [(1,1), (1,5), (5,1), (5,5)]:
        for i in range(0,3):
            for j in range(0,3):
                graph.add_edge((room_x + i, room_y + j, 0), (room_x + i, room_y + j, 1))
                graph.add_edge((room_x + i, room_y + j, 1), (room_x + i, room_y + j, 0))

                graph.add_edge((room_x + i, room_y + j, 1), (room_x + i, room_y + j, 2))
                graph.add_edge((room_x + i, room_y + j, 2), (room_x + i, room_y + j, 1))

                graph.add_edge((room_x + i, room_y + j, 2), (room_x + i, room_y + j, 3))
                graph.add_edge((room_x + i, room_y + j, 3), (room_x + i, room_y + j, 2))

                graph.add_edge((room_x + i, room_y + j, 3), (room_x + i, room_y + j, 0))
                graph.add_edge((room_x + i, room_y + j, 0), (room_x + i, room_y + j, 3))

    ## generate the correct connections/doors between those 4 rooms
    door_up_coords = (4, topology[0] + 1)
    door_down_coords = (4, topology[1] + 5)
    door_left_coords = (topology[2] + 1, 4)
    door_right_coords = (topology[3] + 5, 4)

    graph.add_nodes_from([(door_up_coords[0],door_up_coords[1],dir) for dir in range(4)])
    graph.add_nodes_from([(door_down_coords[0],door_down_coords[1],dir) for dir in range(4)])
    graph.add_nodes_from([(door_left_coords[0],door_left_coords[1],dir) for dir in range(4)])
    graph.add_nodes_from([(door_right_coords[0],door_right_coords[1],dir) for dir in range(4)])

    graph.add_edge((door_up_coords[0],door_up_coords[1],3), (door_up_coords[0],door_up_coords[1],3))
    graph.add_edge((door_up_coords[0],door_up_coords[1],0), (door_up_coords[0]+1,door_up_coords[1],0))
    graph.add_edge((door_up_coords[0]-1,door_up_coords[1],0), (door_up_coords[0],door_up_coords[1],0))
    graph.add_edge((door_up_coords[0],door_up_coords[1],1), (door_up_coords[0],door_up_coords[1],1))
    graph.add_edge((door_up_coords[0],door_up_coords[1],2), (door_up_coords[0]-1,door_up_coords[1],2))
    graph.add_edge((door_up_coords[0]+1,door_up_coords[1],2), (door_up_coords[0],door_up_coords[1],2))

    graph.add_edge((door_down_coords[0],door_down_coords[1],3), (door_down_coords[0],door_down_coords[1],3))
    graph.add_edge((door_down_coords[0],door_down_coords[1],0), (door_down_coords[0]+1,door_down_coords[1],0))
    graph.add_edge((door_down_coords[0]-1,door_down_coords[1],0), (door_down_coords[0],door_down_coords[1],0))
    graph.add_edge((door_down_coords[0],door_down_coords[1],1), (door_down_coords[0],door_down_coords[1],1))
    graph.add_edge((door_down_coords[0],door_down_coords[1],2), (door_down_coords[0]-1,door_down_coords[1],2))
    graph.add_edge((door_down_coords[0]+1,door_down_coords[1],2), (door_down_coords[0],door_down_coords[1],2))

    graph.add_edge((door_left_coords[0],door_left_coords[1],0), (door_left_coords[0],door_left_coords[1],0))
    graph.add_edge((door_left_coords[0],door_left_coords[1],3), (door_left_coords[0],door_left_coords[1]-1,3))
    graph.add_edge((door_left_coords[0],door_left_coords[1]+1,3), (door_left_coords[0],door_left_coords[1],3))
    graph.add_edge((door_left_coords[0],door_left_coords[1],2), (door_left_coords[0],door_left_coords[1],2))
    graph.add_edge((door_left_coords[0],door_left_coords[1],1), (door_left_coords[0],door_left_coords[1]+1,1))
    graph.add_edge((door_left_coords[0],door_left_coords[1]-1,1), (door_left_coords[0],door_left_coords[1],1))

    graph.add_edge((door_right_coords[0],door_right_coords[1],0), (door_right_coords[0],door_right_coords[1],0))
    graph.add_edge((door_right_coords[0],door_right_coords[1],3), (door_right_coords[0],door_right_coords[1]-1,3))
    graph.add_edge((door_right_coords[0],door_right_coords[1]+1,3), (door_right_coords[0],door_right_coords[1],3))
    graph.add_edge((door_right_coords[0],door_right_coords[1],2), (door_right_coords[0],door_right_coords[1],2))
    graph.add_edge((door_right_coords[0],door_right_coords[1],1), (door_right_coords[0],door_right_coords[1]+1,1))
    graph.add_edge((door_right_coords[0],door_right_coords[1]-1,1), (door_right_coords[0],door_right_coords[1],1))

    for x,y in [door_up_coords, door_down_coords, door_left_coords, door_right_coords]:
        graph.add_edge((x,y,0), (x,y,1))
        graph.add_edge((x,y,1), (x,y,0))
        graph.add_edge((x,y,1), (x,y,2))
        graph.add_edge((x,y,2), (x,y,1))
        graph.add_edge((x,y,2), (x,y,3))
        graph.add_edge((x,y,3), (x,y,2))
        graph.add_edge((x,y,3), (x,y,0))
        graph.add_edge((x,y,0), (x,y,3))

    return graph

def find_shortest_path(agent_pos, agent_dir, goal_pos, topology):
    graph = create_maze_graph(topology)

    paths = []
    for dir in range(4):
        # check for path to any of the 4 terminal nodes
        # (corresponding to the goal location with 4 different directions)
        paths.append(nx.shortest_path(graph, source=(*agent_pos, agent_dir), target=(*goal_pos, dir)))

    path_lengths = np.array([len(p)-1 for p in paths])
    shortest_path = paths[np.argmin(path_lengths)]

    return shortest_path

def find_all_shortest_paths(agent_pos, agent_dir, goal_pos, topology):
    graph = create_maze_graph(topology)

    paths = []
    for dir in range(4):
        # check for path to any of the 4 terminal nodes
        # (corresponding to the goal location with 4 different directions)
        for p in nx.all_shortest_paths(graph, source=(*agent_pos, agent_dir), target=(*goal_pos, dir)):
            paths.append(p)

    # not all paths found so far will necessarily have equal length
    path_lengths = np.array([len(p)-1 for p in paths])
    shortest_path_length = np.min(path_lengths)
    shortest_paths = np.array(paths, dtype=object)[np.where(path_lengths == shortest_path_length)[0]]

    return shortest_paths

def find_all_action_values(agent_pos, agent_dir, goal_pos, topology, gamma):
    """
        We will derive the action values (Q values) by enumerating all the actions and finding the state value
        in the corresponding next state. 
    """ 
    graph = create_maze_graph(topology)
    q_values = list()

    # Action 0 : Left (agent_dir decreases)
    paths = []
    for dir in range(4):
        for p in nx.all_shortest_paths(graph, source=(*agent_pos, (agent_dir - 1)%4), target=(*goal_pos, dir)):
            paths.append(p)

    path_lengths = np.array([len(p)-1 for p in paths])
    shortest_path_length = np.min(path_lengths)

    # value of this state is gamma^(path length - 1)
    next_state_value = gamma**(shortest_path_length-1)
    # q_value is 0 + gamma*next_state_value
    q_values.append(gamma*next_state_value)

    # Action 1 : Right (agent_dir increases)
    paths = []
    for dir in range(4):
        for p in nx.all_shortest_paths(graph, source=(*agent_pos, (agent_dir + 1)%4), target=(*goal_pos, dir)):
            paths.append(p)

    path_lengths = np.array([len(p)-1 for p in paths])
    shortest_path_length = np.min(path_lengths)

    # value of this state is gamma^(path length - 1)
    next_state_value = gamma**(shortest_path_length-1)
    # q_value is 0 + gamma*next_state_value
    q_values.append(gamma*next_state_value)

    # Action 2 : Forward (position changes based on agent_dir)
    new_pos = agent_pos
    if agent_dir == 0:
        # looking right
        new_pos = (new_pos[0]+1, new_pos[1])
    if agent_dir == 1:
        # looking down
        new_pos = (new_pos[0], new_pos[1]+1)
    if agent_dir == 2:
        # looking left
        new_pos = (new_pos[0]-1, new_pos[1])
    if agent_dir == 3:
        # looking up
        new_pos = (new_pos[0], new_pos[1]-1)

    # if we moved into the goal, return a value of 1
    if new_pos == goal_pos:
        q_values.append(1)
    else:
        if (*new_pos, 0) not in graph.nodes:
            # we moved into a wall earlier and need to move back
            new_pos = agent_pos

        paths = []
        for dir in range(4):
            for p in nx.all_shortest_paths(graph, source=(*new_pos, agent_dir), target=(*goal_pos, dir)):
                paths.append(p)

        path_lengths = np.array([len(p)-1 for p in paths])
        shortest_path_length = np.min(path_lengths)

        # value of this state is gamma^(path length - 1)
        next_state_value = gamma**(shortest_path_length-1)

        # q_value is 0 + gamma*next_state_value
        q_values.append(gamma*next_state_value)


    return q_values

def does_path_exist(topology, path):
    graph = create_maze_graph(topology)

    return nx.is_simple_path(graph, path)
