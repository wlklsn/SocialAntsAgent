import pickle
import matplotlib.pyplot as plt # type: ignore

plot_our_code = True


if plot_our_code:

    # Define the file paths (make sure these match the paths used in the original script)
    positions_file = "simulations/test/positions.pkl"
    food_areas_positions_file = "simulations/test/food_area_positions.pkl"
    apples_positions_file = "simulations/test/apple_positions.pkl"

    # Load the data

    with open(positions_file, 'rb') as file:
        all_positions = pickle.load(file)

    with open(food_areas_positions_file, 'rb') as file2:
        food_areas_positions = pickle.load(file2)

    with open(apples_positions_file, 'rb') as file3:
        apples_positions = pickle.load(file3)

else: 

    # Define the file paths (make sure these match the paths used in the original script)
    positions_file = "simulations/test/code_basis_positions.pkl"
    food_areas_positions_file = "simulations/test/food_area_positions.pkl"
    apples_positions_file = "simulations/test/apple_positions.pkl"

    with open(positions_file, 'rb') as file:
        all_positions = pickle.load(file)

    with open(food_areas_positions_file, 'rb') as file2:
        food_areas_positions = pickle.load(file2)

    with open(apples_positions_file, 'rb') as file3:
        apples_positions = pickle.load(file3)


# Plot the path of the best agent for each generation
generations = len(all_positions)
arena_length = 600  # Assuming this value from the original simulation


for igen in range(generations):

    if igen == 9:

        print(food_areas_positions[igen])
 

    best_positions = all_positions[igen]
    plt.figure(figsize=(6, 6))

    if plot_our_code:
    

        # Get area positions for this generation
        area_in_gen = food_areas_positions[igen]
        # Get apples positions for this generation
        apples_in_gen = apples_positions[igen]  
        

        # Plot apple areas
        for area in area_in_gen:


            bl_x, bl_y, tr_x, tr_y, apple_count = area

            

                

            box_size = tr_x - bl_x
                       
            
            top_left_x = bl_x
            top_left_y = bl_y + box_size

            rect = plt.Rectangle((bl_x, bl_y), box_size, box_size, linewidth=1, edgecolor='green', facecolor='green', alpha=0.5)
            plt.gca().add_patch(rect)

    # Plot agent path    
    plt.plot(best_positions[:, 0], best_positions[:, 1], marker='o', linestyle='-', color='b')

    plt.title(f'Best Agent Path - Generation {igen+1}')
    plt.xlim(0, arena_length)
    plt.ylim(0, arena_length)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.show()