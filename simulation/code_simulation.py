########################
# Our Imports
########################

import math
import numpy as np
import pickle
from scipy.special import expit 
import time
from smt.sampling_methods import LHS 

# Our Imports

import os

########################
# Functions
########################

def ensure_path_exists(_directory):

    if not os.path.exists(_directory):

        print("\n------ ATTENTION, PLEASE READ! ------ \n")

        print("Folder does not exist, creating folder for simulation data...\n")

        os.makedirs(_directory)

        
        print("Sorry, I set up the git so that the sim data is one directory upwards from the files itself!\n")
        print("If the directory is created in a folder that you you would prefer not to, just press 'CTRL + C' to stop the current simulation and delete the folder at:\n" + _directory + "\n")
        print("After that move the 'code_simulation.py' into an additional folder in the directory it is currently in or ask me about it!\n")
        print("------ ATTENTION, PLEASE READ FROM THE BEGINING! ------ \n")

def save_data(_file_paths, _data):

    ensure_path_exists(_file_paths["directory"])
    
    print("Saving data...")
    # Save data
    with open(_file_paths["brains"], 'wb') as file:
        pickle.dump(_data["brains"], file)

    with open(_file_paths["scores"], 'wb') as file2:
        pickle.dump(_data["scores"], file2)

    with open(_file_paths["metadata"], 'wb') as file3:
        pickle.dump(_data["metadata"], file3)


    with open(_file_paths["positions"], 'wb') as file4:
        pickle.dump(_data["positions"], file4)

    with open(_file_paths["food_area_positions"], 'wb') as file5:
        pickle.dump(_data["food_area_positions"], file5)

    with open(_file_paths["apple_positions"], 'wb') as file6:
        pickle.dump(_data["apple_positions"], file6)

    print("Saving data finished. You can now execute the visualisation!\n")

def turn_and_speed(o_steer, o_speed, st_incr, min_speed, max_speed):

    coef_steer = o_steer - 0.5
    dsteering = coef_steer * st_incr
    speed = min_speed + (o_speed * (max_speed - min_speed))  # Normalize speed
    return dsteering, speed


def mutate(size_vec, _rng, prob_small=.33, prob_large=.01, sd_small=0.5, sd_large=5):
    small_mutations = np.multiply(_rng.choice(a=[0, 1], size=size_vec, replace=True, p=[1-prob_small, prob_small]),
                _rng.normal(loc=0.0, scale=sd_small, size=size_vec))
    large_mutations = np.multiply(_rng.choice(a=[0, 1], size=size_vec, replace=True, p=[1-prob_large, prob_large]),
                _rng.normal(loc=0.0, scale=sd_large, size=size_vec))
    mutations_array = small_mutations + large_mutations
    return mutations_array


def input_closestwall(_x, _y, _headings, _world_length, n):
    dist_walls = np.zeros(shape=(n, 4))
    angle_walls = np.zeros(shape=(n, 4))
    output = np.zeros(shape=(n, 2))

    # calculate distance from this fish to the closest point of each wall
    # 0, 1, 2, 3 resp. for south, east, north and west
    dist_walls[:, 0] = _y  # bas
    dist_walls[:, 1] = _world_length - _x  # droite
    dist_walls[:, 2] = _world_length - _y  # haut
    dist_walls[:, 3] = _x  # gauche

    # calculate angle between this fish heading and each wall
    angle_walls[:, 0] = (_headings + math.pi / 2)
    angle_walls[:, 1] = _headings
    angle_walls[:, 2] = (_headings - math.pi / 2)
    angle_walls[:, 3] = (_headings - math.pi)

    # normalise and center the inputs (0: distance to closest wall; 1: angle to wall)
    closest_wall = dist_walls.argmin(axis=1).astype('int64')
    output[:, 0] = dist_walls[np.arange(n), closest_wall]
    output[:, 1] = angle_walls[np.arange(n), closest_wall]
    
    

    return output

######################
#speed output node
######################

def find_bins(_x, _arena_interval):
    bins = np.floor(_x / _arena_interval).astype('int64')
    return bins


def generate_brain_params(_nagents, _in_nodes, _hid_nodes, _out_nodes, _low_limit, _top_limit):
    nwih = _in_nodes * _hid_nodes
    nwho = _hid_nodes * _out_nodes
    ninputs = nwih + nwho + _hid_nodes + _out_nodes

    xlimits = np.zeros(shape=(ninputs, 2))
    xlimits[:, [0, 1]] = [_low_limit, _top_limit]
    sampling = LHS(xlimits=xlimits, criterion='maximin')
    params = sampling(_nagents)

    twih = params[:, range(nwih)]
    twho = params[:, range(nwih, nwih + nwho)]
    tbh = params[:, range(nwih + nwho, nwih + nwho + _hid_nodes)]
    tbo = params[:, range(nwih + nwho + _hid_nodes, nwih + nwho + _hid_nodes + _out_nodes)]

    output = [twih.reshape((_nagents, _in_nodes, _hid_nodes)), twho.reshape((_nagents, _hid_nodes, _out_nodes)),
            tbh.reshape((_nagents, 1, _hid_nodes)), tbo.reshape((_nagents, 1, _out_nodes))]
    return output

def is_valid_area(_new_area, _occupied_area):
    """
    Check if the new rectangle intersects with any of the old rectangles.

    Parameters:
    new_area (tuple): A tuple of four integers (x1, y1, x2, y2) representing the new rectangle.
    occupied_area (list): A list of tuples, each containing four integers (x1, y1, x2, y2) representing the old rectangles.

    Returns:
    bool: True if the new rectangle does not intersect with any old rectangles, False otherwise.
    """
    new_x_bl, new_y_bl, new_x_tr, new_y_tr = _new_area

    for old_x_bl, old_y_bl, old_x_tr, old_y_tr in _occupied_area:
        # Check if the new rectangle does not intersect with the old rectangle
        if not (new_x_tr <= old_x_bl or new_x_bl >= old_x_tr or new_y_tr <= old_y_bl or new_y_bl >= old_y_tr):
            return False

    return True

#TODO: Check first if food areas will take too much space
def generate_food_area(_world_size, _area_number, _area_size, _dev_area_size, _max_apples):

    area_positions = np.zeros((_area_number, 5)) 

    occupied_area = []
    area = 0

    attempts = 0

    while area < _area_number and attempts < 1000:
        attempts += 1
        
        deviation = np.random.randint(0, _dev_area_size)
        
        bottom_left_x = np.random.randint((_world_size/_area_number) * area, (_world_size/_area_number) * (area + 1) - (_area_size + _dev_area_size))
        bottom_left_y = np.random.randint((_world_size/_area_number) * area, (_world_size/_area_number) * (area + 1) - (_area_size + _dev_area_size))

        top_right_x = bottom_left_x + _area_size + deviation
        top_right_y = bottom_left_y + _area_size + deviation

        new_area = (bottom_left_x, bottom_left_y, top_right_x, top_right_y)

        if is_valid_area(new_area, occupied_area):
            
            occupied_area.append(new_area)

            area_positions[area][0] = bottom_left_x
            area_positions[area][1] = bottom_left_y

            area_positions[area][2] = top_right_x
            area_positions[area][3] = top_right_y

            numb_apples = np.random.randint(1,_max_apples)

            area_positions[area][4] = numb_apples
            #area_positions[area][4] = 1

            attempts = 0
            area += 1

    return area_positions

def generate_apples(_area_number, _food_area_ini, _max_apples):

    apple_ini = np.zeros(shape=(_area_number, _max_apples, 2))

    for area in range(_area_number):

        napples = _food_area_ini[area][4].astype(int)

        for iapple in range(napples):

            bl_x = _food_area_ini[area][0]
            tr_x = _food_area_ini[area][2]

            apple_x = np.random.randint(bl_x, tr_x)
            
            apple_ini[area][iapple][0] = apple_x

            bl_y = _food_area_ini[area][1]
            tr_y = _food_area_ini[area][3]

            apple_y = np.random.randint(bl_y, tr_y)
            
            apple_ini[area][iapple][1] = apple_y

    return apple_ini

def collect_apple(_food_area_positions, _area_number, _apple_ini, _agents_x, _agents_y, _agents_ingame, _apple_radius, _scores, _apple_points):
    
    for agent in np.where(_agents_ingame)[0]:
                
        for area in range(_area_number):

            current_apple = _food_area_positions[agent][area][4].astype(int) - 1

        
            if current_apple != -1:

                apple_x = _apple_ini[area][current_apple][0]
                apple_y = _apple_ini[area][current_apple][1]

                if np.linalg.norm(np.array([apple_x, apple_y]) - np.array([_agents_x[agent], _agents_y[agent]])) < _apple_radius:

                    _scores[agent] += _apple_points
                    _food_area_positions[agent][area][4]  -= 1



def is_inside_vectorized(x, y, food_areas):
    # Initialize a boolean array to keep track of agents inside any food area
    inside = np.zeros((x.size, food_areas.shape[1]), dtype=bool)
    # Existing logic in is_inside_vectorized function
    for i in range(food_areas.shape[1]):
        x_min, y_min, x_max, y_max = food_areas[:, i, :4].T
      
        inside[:, i] = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    # If the agent is inside any food area, set the corresponding entry to True
    return inside.any(axis=1).astype(int)
    
########################
# Variables
########################

tic = time.perf_counter()

rng = np.random.default_rng()
in_nodes = 3
hid_nodes = 3
out_nodes = 2
nagents = 1000         # default: 10000
agent_life = 50
generations = 20        # default: 20 gens
min_speed = 20
max_speed = 100
angle_increment = 6.28  # twice the actual maximum angle turned (see func turn())
world_length = 600
max_dist = world_length / 2
arena_bins = 20
arena_interval = world_length / arena_bins
min_ini_brain_value = -50
max_ini_brain_value = 50


# New Values

apple_points = 300
apple_radius = 10

max_apples = 3
area_number = 2

area_size = 75
dev_area_size = 10

########################
# Initialisation
########################

all_scores = np.zeros(shape=(generations, nagents))
all_brains = []  # best brain of each generation
all_positions = []  # to store positions of the best agent of each generation
all_food_area_positions = []
all_apples_positions = []


inputs = np.zeros(shape=(nagents, 1, in_nodes))

brain_params = generate_brain_params(nagents, in_nodes, hid_nodes, out_nodes, min_ini_brain_value, max_ini_brain_value)

weights_ih = brain_params[0]
weights_ho = brain_params[1]

bias_h = brain_params[2]
bias_o = brain_params[3]

# (first initial condition is always from the centre of the arena to compare scores across generations -- not taken into
# account for fitness calculation) ---> not necessary
headings_ini = [0., 0., -math.pi, -math.pi / 2, math.pi / 2]
n_headini = len(headings_ini)
n_headini_r = n_headini - 1
x_ini_nofit = np.array([max_dist])
y_ini_nofit = np.array([max_dist])

metadata = {
    "speed": speed,
    "worldSize": world_length,
    "totalPopulation": nagents,
    "genNb": generations,
    "neuronsHL": hid_nodes,
    "maxAngularIncr": angle_increment,
    "InitialConditionsNb": n_headini_r,
    "MinIniBrainValue": min_ini_brain_value,
    "MaxIniBrainValue": max_ini_brain_value
}

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the folder one directory upwards
output_dir = os.path.abspath(os.path.join(current_dir, '..', 'simulation_data'))

file_paths = {
    "directory": output_dir,
    "brains": output_dir + "/brains.pkl",
    "scores": output_dir + "/scores.pkl",
    "metadata": output_dir + "/metadata.pkl",
    "positions": output_dir + "/positions.pkl",
    "food_area_positions": output_dir + "/food_area_positions.pkl",
    "apple_positions": output_dir + "/apple_positions.pkl"
}

ensure_path_exists(file_paths["directory"])

########################
# Calculation
########################

for igen in range(generations):
    print("Generation: ", igen)
    x_ini_r = np.full(fill_value=rng.uniform(low=0, high=world_length, size=1), shape=n_headini_r)
    y_ini_r = np.full(fill_value=rng.uniform(low=0, high=world_length, size=1), shape=n_headini_r)
    x_ini = np.concatenate((x_ini_nofit, x_ini_r))
    y_ini = np.concatenate((y_ini_nofit, y_ini_r))

    scores = np.zeros(shape=nagents)

    # to store positions of each agent at each step
    positions = np.zeros((nagents, agent_life, 2)) 

    # initilize food area parameters
    food_area_ini = generate_food_area(world_length, area_number, area_size, dev_area_size, max_apples)

    # initilize apple parameters
    apple_ini = generate_apples(area_number, food_area_ini, max_apples)

    # to store positions of each apple at each step
    apple_positions = np.ones((nagents, agent_life, area_number, 2)) * -1
      

    for iic in range(n_headini):
        #print(iic)
        exploration_mat = np.zeros(shape=(nagents, arena_bins, arena_bins))
        agents_ingame = np.full(shape=nagents, fill_value=True)
        x = np.full(shape=nagents, fill_value=x_ini[iic])
        y = np.full(shape=nagents, fill_value=y_ini[iic])
        headings = np.full(shape=nagents, fill_value=headings_ini[iic])

        x_bin = np.floor(x / arena_interval).astype(int)
        y_bin = np.floor(y / arena_interval).astype(int)
        exploration_mat[:, y_bin, x_bin] = 1

        # to store positions of each food_area at each step
        food_area_positions = np.tile(food_area_ini, (nagents, 1, 1))


        istep = 1
        n_ingame = agents_ingame.sum()

        while n_ingame > 0 and istep <= agent_life:
            
            
            collect_apple(food_area_positions, area_number, apple_ini, x, y, agents_ingame, apple_radius, scores, apple_points)
            
            #store positions of apples
            if iic == n_headini - 1:
                
                """
                # Iterate only over active agents
                active_agent_indices = np.where(agents_ingame)[0]

                # Extract the count of apples remaining for all active agents and areas
                current_apples = food_area_positions[active_agent_indices, :, 4].astype(int) - 1

                # Ensure that current_apples shape matches for broadcasting
                # Reshape current_apples to have the same leading dimensions as apple_positions
                reshaped_current_apples = current_apples[:, :, np.newaxis]

                # Update apple positions for all active agents and areas
                for i, agent_idx in enumerate(active_agent_indices):
                    apple_positions[agent_idx, istep-1, :, :] = apple_ini[np.arange(area_number), reshaped_current_apples[i].squeeze(), :]

                """
                # Iterate only over active agents
                active_agent_indices = np.where(agents_ingame)[0]

                # Iterate through each agent
                for agent_idx in active_agent_indices:

                    # Iterate through each food area for the agent
                    for area_idx in range(area_number):
                        
                        # Assuming the last value in the third dimension is the count of apples remaining
                        current_apple = int(food_area_positions[agent_idx, area_idx, 4]) - 1 
 
                        apple_positions[agent_idx, istep-1, area_idx, :] = apple_ini[area_idx, current_apple, :]
                
            # set the angles in [-pi, pi]
            headings[agents_ingame] = np.arctan2(np.sin(headings[agents_ingame]),
                                                 np.cos(headings[agents_ingame]))

            # calculate distance from this fish to the closest point of each wall
            both_inputs = input_closestwall(x[agents_ingame], y[agents_ingame], headings[agents_ingame], world_length, n_ingame)
            # normalise and center the inputs (0: distance to closest wall; 1: angle to wall)
            inputs[agents_ingame, 0, 0] = both_inputs[:, 0] / max_dist
            inputs[agents_ingame, 0, 1] = (both_inputs[:, 1] % (2 * math.pi)) / (2 * math.pi)
            inputs[agents_ingame, 0, 2] = is_inside_vectorized(x[agents_ingame], y[agents_ingame], food_area_positions[agents_ingame])
               
            # move / restric calculations to agents_ingame
            temp_h = expit(np.matmul(inputs[agents_ingame, :, :], weights_ih[agents_ingame, :, :]) + bias_h[agents_ingame, :, :])
            temp_o = expit(np.matmul(temp_h, weights_ho[agents_ingame, :, :]) + bias_o[agents_ingame, :, :])

            delta_phi, speed = turn_and_speed(temp_o[:,0 , 0],temp_o[:,0, 1], angle_increment, min_speed, max_speed)
            headings[agents_ingame] += delta_phi
            x[agents_ingame] += (speed * np.cos(headings[agents_ingame]))
            y[agents_ingame] += (speed * np.sin(headings[agents_ingame]))

            # store positions of agents
            positions[agents_ingame, istep-1, 0] = x[agents_ingame]
            positions[agents_ingame, istep-1, 1] = y[agents_ingame]

            # evaluate score for each agent
            agents_ingame = (x <= world_length) & (x >= 0) & (y <= world_length) & (y >= 0)

            x_bin = find_bins(x[agents_ingame], arena_interval)
            y_bin = find_bins(y[agents_ingame], arena_interval)
            exploration_mat[agents_ingame, y_bin, x_bin] = 1

            istep += 1
            n_ingame = agents_ingame.sum()


        scores += (exploration_mat.sum(axis=1)).sum(axis=1)

    # calculate score & fitness
    score_sum = scores.sum()
    score_max = scores.max()
    score_av = scores.mean()
    

    if(score_sum != 0):

        fitness = scores / score_sum
    else:

        fitness = np.zeros_like(scores)

    new_pop = rng.choice(a=range(nagents), size=nagents, replace=True, p=fitness)
    all_scores[igen, :] = scores
    ibest = scores.argmax()
    all_brains.append([weights_ih[ibest, :, :], weights_ho[ibest, :, :], bias_h[ibest, :, :], bias_o[ibest, :, :]])

    # store positions of the best agent in this generation
    best_positions = positions[ibest, :, :]
    all_positions.append(best_positions)

    # store positions of the food_areas in this generation
    best_food_area_positions = food_area_positions[ibest, :, :]
    all_food_area_positions.append(best_food_area_positions)

    # store positions of the apples of best agents in this generation
    best_apples = apple_positions[ibest, :, :, :]
    all_apples_positions.append(best_apples)   

    # weigths and biases
    weights_ih = weights_ih[new_pop, :, :] + mutate((nagents, in_nodes, hid_nodes), _rng=rng)
    weights_ho = weights_ho[new_pop, :, :] + mutate((nagents, hid_nodes, out_nodes), _rng=rng)
    bias_h = bias_h[new_pop, :, :] + mutate((nagents, 1, hid_nodes), _rng=rng)
    bias_o = bias_o[new_pop, :, :] + mutate((nagents, 1, out_nodes), _rng=rng)

    print(score_av)
    print(score_max)
    print()

toc = time.perf_counter()
print(f"Simulation executed in {toc - tic:0.4f} seconds")
print(all_scores.mean(axis=1))
print(all_scores.max(axis=1))

########################
# Saving Data
########################

# Prepare data for saving 
data = {
    "brains": all_brains,
    "scores": all_scores,
    "metadata": metadata,
    "positions": all_positions,
    "food_area_positions": all_food_area_positions,
    "apple_positions":all_apples_positions
}

# Save data
save_data(file_paths, data)
