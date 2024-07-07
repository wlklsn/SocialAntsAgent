import math
import numpy as np
import pickle
from scipy.special import expit
import time
from smt.sampling_methods import LHS
import os

def ensure_path_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(_file_paths, _data):
    ensure_path_exists(_file_paths["directory"])
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

    dist_walls[:, 0] = _y
    dist_walls[:, 1] = _world_length - _x
    dist_walls[:, 2] = _world_length - _y
    dist_walls[:, 3] = _x

    angle_walls[:, 0] = (_headings + math.pi / 2)
    angle_walls[:, 1] = _headings
    angle_walls[:, 2] = (_headings - math.pi / 2)
    angle_walls[:, 3] = (_headings - math.pi)

    closest_wall = dist_walls.argmin(axis=1).astype('int64')
    output[:, 0] = dist_walls[np.arange(n), closest_wall]
    output[:, 1] = angle_walls[np.arange(n), closest_wall]

    return output

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

def is_valid_area(_occupied_area, _new_bottom_left, _new_top_right, _world_size, _area_size, _dev_area_size):
    new_x_bl, new_y_bl = _new_bottom_left
    new_x_tr, new_y_tr = _new_top_right

    if new_x_bl <= 0 and new_y_bl <= 0 and new_x_tr >= _area_size and new_y_tr >= _world_size:
        return False

    if len(_occupied_area) == 0:
        return True

    for (bl, tr) in _occupied_area:
        x_bl, y_bl = bl
        if new_x_bl >= x_bl - (_area_size + _dev_area_size) and new_x_bl <= x_bl + (_area_size + _dev_area_size):
            if new_y_bl >= y_bl - (_area_size + _dev_area_size) and new_y_bl <= y_bl + (_area_size + _dev_area_size):
                return False

    return True

def generate_food_area(_world_size, _area_number, _area_size, _dev_area_size, _max_apples):
    area_positions = np.zeros((_area_number, 5)) 
    occupied_area = []
    area = 0

    while area < _area_number:
        deviation = np.random.randint(0, _dev_area_size)
        bottom_left_x = np.random.randint((_world_size/_area_number) * area, (_world_size/_area_number) * (area + 1) - (_area_size + _dev_area_size))
        bottom_left_y = np.random.randint((_world_size/_area_number) * area, (_world_size/_area_number) * (area + 1) - (_area_size + _dev_area_size))

        bottom_left = (bottom_left_x, bottom_left_y)
        top_right_x = bottom_left_x + _area_size + deviation
        top_right_y = bottom_left_y + _area_size + deviation

        top_right = (top_right_x, top_right_y)
        temp = is_valid_area(occupied_area, bottom_left, top_right, _world_size, _area_size, _dev_area_size)

        if temp:
            occupied_area.append((bottom_left, top_right))
            area_positions[area][0] = bottom_left_x
            area_positions[area][1] = bottom_left_y
            area_positions[area][2] = top_right_x
            area_positions[area][3] = top_right_y
            numb_apples = np.random.randint(1,_max_apples)
            area_positions[area][4] = numb_apples
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

def collect_apple(_food_area_positions, _area_number, _apple_ini, _agents_x, _agents_y, _agents_ingame, _apple_radius, _scores, _igen, _iic, _n_headini):
    for agent in np.where(_agents_ingame)[0]:
        for area in range(_area_number):
            current_apple = _food_area_positions[agent][area][4].astype(int) - 1
            if current_apple != -1:
                apple_x = _apple_ini[area][current_apple][0]
                apple_y = _apple_ini[area][current_apple][1]
                if np.linalg.norm(np.array([apple_x, apple_y]) - np.array([_agents_x[agent], _agents_y[agent]])) < _apple_radius:
                    _scores[agent] += apple_points
                    _food_area_positions[agent][area][4]  -= 1

def is_inside_vectorized(x, y, food_areas):
    inside = np.zeros((x.size, food_areas.shape[1]), dtype=bool)
    for i in range(food_areas.shape[1]):
        x_min, y_min, x_max, y_max = food_areas[:, i, :4].T
        inside[:, i] = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    return inside.any(axis=1).astype(int)

tic = time.perf_counter()
rng = np.random.default_rng()
in_nodes = 3
hid_nodes = 3
out_nodes = 2  # Updated: One output for steering, one for speed
nagents = 1000
agent_life = 50
generations = 20
min_speed = 20  # Updated: Minimum speed for agents
max_speed = 100  # Updated: Maximum speed for agents
angle_increment = 6.28
world_length = 600
max_dist = world_length / 2
arena_bins = 20
arena_interval = world_length / arena_bins
min_ini_brain_value = -50
max_ini_brain_value = 50

apple_points = 300
apple_radius = 10
max_apples = 3
area_number = 2
area_size = 75
dev_area_size = 10

all_scores = np.zeros(shape=(generations, nagents))
all_brains = []
all_positions = []
all_food_area_positions = []
all_apples_positions = []

inputs = np.zeros(shape=(nagents, 1, in_nodes))
brain_params = generate_brain_params(nagents, in_nodes, hid_nodes, out_nodes, min_ini_brain_value, max_ini_brain_value)

weights_ih = brain_params[0]
weights_ho = brain_params[1]
bias_h = brain_params[2]
bias_o = brain_params[3]

headings_ini = [0., 0., -math.pi, -math.pi / 2, math.pi / 2]
n_headini = len(headings_ini)
n_headini_r = n_headini - 1
x_ini_nofit = np.array([max_dist])
y_ini_nofit = np.array([max_dist])

metadata = {
    "speed": max_speed,  # Updated: Max speed included in metadata
    "worldSize": world_length,
    "totalPopulation": nagents,
    "genNb": generations,
    "neuronsHL": hid_nodes,
    "maxAngularIncr": angle_increment,
    "InitialConditionsNb": n_headini_r,
    "MinIniBrainValue": min_ini_brain_value,
    "MaxIniBrainValue": max_ini_brain_value
}

output_dir = "results/"

file_paths = {
    "directory": output_dir,
    "brains": output_dir + "brains.pkl",
    "scores": output_dir + "scores.pkl",
    "metadata": output_dir + "metadata.pkl",
    "positions": output_dir + "positions.pkl",
    "food_area_positions": output_dir + "food_area_positions.pkl",
    "apple_positions": output_dir + "apple_positions.pkl"
}

for igen in range(generations):
    print("Generation: ", igen)
    x_ini_r = np.full(fill_value=rng.uniform(low=0, high=world_length, size=1), shape=n_headini_r)
    y_ini_r = np.full(fill_value=rng.uniform(low=0, high=world_length, size=1), shape=n_headini_r)
    x_ini = np.concatenate((x_ini_nofit, x_ini_r))
    y_ini = np.concatenate((y_ini_nofit, y_ini_r))

    scores = np.zeros(shape=nagents)
    positions = np.zeros((nagents, agent_life, 2))
    food_area_ini = generate_food_area(world_length, area_number, area_size, dev_area_size, max_apples)
    food_area_positions = np.tile(food_area_ini, (nagents, 1, 1))
    apple_ini = generate_apples(area_number, food_area_ini, max_apples)
    apple_positions = np.ones((nagents, agent_life, max_apples, 2)) * -1

    for iic in range(n_headini):
        exploration_mat = np.zeros(shape=(nagents, arena_bins, arena_bins))
        agents_ingame = np.full(shape=nagents, fill_value=True)
        x = np.full(shape=nagents, fill_value=x_ini[iic])
        y = np.full(shape=nagents, fill_value=y_ini[iic])
        headings = np.full(shape=nagents, fill_value=headings_ini[iic])

        x_bin = np.floor(x / arena_interval).astype(int)
        y_bin = np.floor(y / arena_interval).astype(int)
        exploration_mat[:, y_bin, x_bin] = 1

        istep = 1
        n_ingame = agents_ingame.sum()

        while n_ingame > 0 and istep <= agent_life:
            collect_apple(food_area_positions, area_number, apple_ini, x, y, agents_ingame, apple_radius, scores, igen, iic, n_headini)
            if iic == n_headini - 1:
                apples_ingame = food_area_positions[agents_ingame]
                current_apple = apples_ingame[:,:,4]
                current_apple = current_apple.astype(int) - 1
                valid_mask = current_apple >= 0

                for arena in range(area_number):
                    for i in range(current_apple.shape[0]):
                        if agents_ingame[i]:
                            if valid_mask[i, arena]:
                                temp1 = apple_ini[arena, current_apple[i, arena], 0]
                                apple_positions[i, istep - 1, arena, 0] = temp1
                                temp2 = apple_ini[arena, current_apple[i, arena], 1]
                                apple_positions[i, istep - 1, arena, 1] = temp2
                            else:
                                apple_positions[i, istep - 1, arena, 0] = -1
                                apple_positions[i, istep - 1, arena, 1] = -1

            headings[agents_ingame] = np.arctan2(np.sin(headings[agents_ingame]), np.cos(headings[agents_ingame]))
            both_inputs = input_closestwall(x[agents_ingame], y[agents_ingame], headings[agents_ingame], world_length, n_ingame)
            inputs[agents_ingame, 0, 0] = both_inputs[:, 0] / max_dist
            inputs[agents_ingame, 0, 1] = (both_inputs[:, 1] % (2 * math.pi)) / (2 * math.pi)
            inputs[agents_ingame, 0, 2] = is_inside_vectorized(x[agents_ingame], y[agents_ingame], food_area_positions[agents_ingame])
            
            temp_h = expit(np.matmul(inputs[agents_ingame, :, :], weights_ih[agents_ingame, :, :]) + bias_h[agents_ingame, :, :])
            temp_o = expit(np.matmul(temp_h, weights_ho[agents_ingame, :, :]) + bias_o[agents_ingame, :, :])

            delta_phi, speed = turn_and_speed(temp_o[:, 0, 0], temp_o[:, 0, 1], angle_increment, min_speed, max_speed)  # Updated: get both steering and speed
            headings[agents_ingame] += delta_phi
            x[agents_ingame] += (speed * np.cos(headings[agents_ingame]))
            y[agents_ingame] += (speed * np.sin(headings[agents_ingame]))

            positions[agents_ingame, istep-1, 0] = x[agents_ingame]
            positions[agents_ingame, istep-1, 1] = y[agents_ingame]

            agents_ingame = (x <= world_length) & (x >= 0) & (y <= world_length) & (y >= 0)

            x_bin = find_bins(x[agents_ingame], arena_interval)
            y_bin = find_bins(y[agents_ingame], arena_interval)
            exploration_mat[agents_ingame, y_bin, x_bin] = 1

            istep += 1
            n_ingame = agents_ingame.sum()

        scores += (exploration_mat.sum(axis=1)).sum(axis=1)

    score_sum = scores.sum()
    score_max = scores.max()
    score_av = scores.mean()

    fitness = scores / score_sum
    new_pop = rng.choice(a=range(nagents), size=nagents, replace=True, p=fitness)
    all_scores[igen, :] = scores
    ibest = scores.argmax()
    all_brains.append([weights_ih[ibest, :, :], weights_ho[ibest, :, :], bias_h[ibest, :, :], bias_o[ibest, :, :]])

    
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
