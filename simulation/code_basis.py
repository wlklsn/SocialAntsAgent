import math
import numpy as np
import pickle
from scipy.special import expit # type: ignore
import time
from smt.sampling_methods import LHS # type: ignore


def turn(o_steer, st_incr):
    coef_steer = o_steer - 0.5
    dsteering = coef_steer * st_incr
    return dsteering


def mutate(size_vec, _rng, prob_small=.33, prob_large=.01, sd_small=0.5, sd_large=5):
    small_mutations = np.multiply(_rng.choice(a=[0, 1], size=size_vec, replace=True, p=[1-prob_small, prob_small]),
                _rng.normal(loc=0.0, scale=sd_small, size=size_vec))
    large_mutations = np.multiply(_rng.choice(a=[0, 1], size=size_vec, replace=True, p=[1-prob_large, prob_large]),
                _rng.normal(loc=0.0, scale=sd_large, size=size_vec))
    mutations_array = small_mutations + large_mutations
    return mutations_array


def input_closestwall(_x, _y, _headings, _arena_length, n):
    dist_walls = np.zeros(shape=(n, 4))
    angle_walls = np.zeros(shape=(n, 4))
    output = np.zeros(shape=(n, 2))

    # calculate distance from this fish to the closest point of each wall
    # 0, 1, 2, 3 resp. for south, east, north and west
    dist_walls[:, 0] = _y  # bas
    dist_walls[:, 1] = _arena_length - _x  # droite
    dist_walls[:, 2] = _arena_length - _y  # haut
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


tic = time.perf_counter()

rng = np.random.default_rng()
in_nodes = 2
hid_nodes = 3
out_nodes = 1
nagents = 10000
agent_life = 50
generations = 20
speed = 50
angle_increment = 6.28  # twice the actual maximum angle turned (see func turn())
arena_length = 600
max_dist = arena_length / 2
arena_bins = 20
arena_interval = arena_length / arena_bins
min_ini_brain_value = -50
max_ini_brain_value = 50

# initialisation
all_scores = np.zeros(shape=(generations, nagents))
all_brains = []  # best brain of each generation
all_positions = []  # to store positions of the best agent of each generation

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
    "arenaSize": arena_length,
    "totalPopulation": nagents,
    "genNb": generations,
    "neuronsHL": hid_nodes,
    "maxAngularIncr": angle_increment,
    "InitialConditionsNb": n_headini_r,
    "MinIniBrainValue": min_ini_brain_value,
    "MaxIniBrainValue": max_ini_brain_value
}

output_dir = "simulations/test/"

str_brains = output_dir + "code_basis_" +"brains.pkl"
str_scores = output_dir + "code_basis_" + "scores.pkl"
str_md = output_dir + "code_basis_" + "metadata.pkl"
str_positions = output_dir + "code_basis_" + "positions.pkl"

# calculation
for igen in range(generations):
    print(igen)
    x_ini_r = np.full(fill_value=rng.uniform(low=0, high=arena_length, size=1), shape=n_headini_r)
    y_ini_r = np.full(fill_value=rng.uniform(low=0, high=arena_length, size=1), shape=n_headini_r)
    x_ini = np.concatenate((x_ini_nofit, x_ini_r))
    y_ini = np.concatenate((y_ini_nofit, y_ini_r))

    scores = np.zeros(shape=nagents)
    positions = np.zeros((nagents, agent_life, 2))  # to store positions of each agent at each step

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
            # set the angles in [-pi, pi]
            headings[agents_ingame] = np.arctan2(np.sin(headings[agents_ingame]),
                                                 np.cos(headings[agents_ingame]))

            # calculate distance from this fish to the closest point of each wall
            both_inputs = input_closestwall(x[agents_ingame], y[agents_ingame], headings[agents_ingame], arena_length, n_ingame)
            # normalise and center the inputs (0: distance to closest wall; 1: angle to wall)
            inputs[agents_ingame, 0, 0] = both_inputs[:, 0] / max_dist
            inputs[agents_ingame, 0, 1] = (both_inputs[:, 1] % (2 * math.pi)) / (2 * math.pi)

            # move / restric calculations to agents_ingame
            temp_h = expit(np.matmul(inputs[agents_ingame, :, :], weights_ih[agents_ingame, :, :]) + bias_h[agents_ingame, :, :])
            temp_o = expit(np.matmul(temp_h, weights_ho[agents_ingame, :, :]) + bias_o[agents_ingame, :, :])

            delta_phi = turn(temp_o, angle_increment).reshape(n_ingame)
            headings[agents_ingame] += delta_phi
            x[agents_ingame] += (speed * np.cos(headings[agents_ingame]))
            y[agents_ingame] += (speed * np.sin(headings[agents_ingame]))

            # store positions of agents
            positions[agents_ingame, istep-1, 0] = x[agents_ingame]
            positions[agents_ingame, istep-1, 1] = y[agents_ingame]

            # evaluate score for each agent
            agents_ingame = (x <= arena_length) & (x >= 0) & (y <= arena_length) & (y >= 0)

            x_bin = find_bins(x[agents_ingame], arena_interval)
            y_bin = find_bins(y[agents_ingame], arena_interval)
            # exploration_mat[agents_ingame, y_bin, x_bin] += 1  # uncomment to goal: survive
            exploration_mat[agents_ingame, y_bin, x_bin] = 1  # comment to stop goal: explore
            
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

    # store positions of the best agent
    best_positions = positions[ibest, :, :]
    all_positions.append(best_positions)

    weights_ih = weights_ih[new_pop, :, :] + mutate((nagents, in_nodes, hid_nodes), _rng=rng)
    weights_ho = weights_ho[new_pop, :, :] + mutate((nagents, hid_nodes, out_nodes), _rng=rng)
    bias_h = bias_h[new_pop, :, :] + mutate((nagents, 1, hid_nodes), _rng=rng)
    bias_o = bias_o[new_pop, :, :] + mutate((nagents, 1, out_nodes), _rng=rng)

    print(score_av)

toc = time.perf_counter()
print(f"Simulation executed in {toc - tic:0.4f} seconds")
print(all_scores.mean(axis=1))
print(all_scores.max(axis=1))

with open(str_brains, 'wb') as file:
    pickle.dump(all_brains, file)

with open(str_scores, 'wb') as file2:
    pickle.dump(all_scores, file2)

with open(str_md, 'wb') as file3:
    pickle.dump(metadata, file3)

with open(str_positions, 'wb') as file4:
    pickle.dump(all_positions, file4)