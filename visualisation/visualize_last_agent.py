import cv2
import numpy as np
import pickle

# Boolean which code to simulate

plot_our_code = True

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

if plot_our_code:

    # Load the apple area centers and box sizes
    with open(file_paths["food_area_positions"], 'rb') as file:
        apple_areas_info = pickle.load(file)

    # Load the saved apple positions for the last generation and the best agent
    with open(file_paths["apple_positions"], 'rb') as file:
        all_apple_positions = pickle.load(file)

else: 
    # Define the file paths
    positions_file = "result/code_basis_positions.pkl"

# Load the positions data
with open(file_paths["positions"], 'rb') as file:
    all_positions = pickle.load(file)



################ TESTING ##################



################ TESTING ##################


################ Variables ##################



# Define arena and apple area coordinates
arena_bottom_left = (0, 0)
arena_top_right = (600, 600)

# Define apple properties
apple_radius = 10

# Define video properties
frame_width = 600
frame_height = 600
fps = 5

# Define colors
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
black = (0, 0, 0)
white = (255, 255, 255)

################ TESTING ##################

# Define the video output file path
video_output = "simulations/test/agent_path_and_apple_collection.mp4"

# Create video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output, fourcc, fps, (frame_width, frame_height))


# Define last generation
last_generation_positions = all_positions[-1]

if plot_our_code:
    last_gen_arena = apple_areas_info[-1]
    last_gen_apples = all_apple_positions[-1]

################ TESTING #################

#print(len(last_gen_apples))

#print(len(last_generation_positions))

#print()
#print()
#print()

################ TESTING ##################


#print(last_gen_apples.shape)
#print(last_gen_apples)

print(last_gen_arena)

for life_steps in range(len(last_generation_positions)):
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255 # Create white background
    
    # Draw arena boundary
    cv2.rectangle(frame, arena_bottom_left, arena_top_right, black, 2)

    # Draw green areas
    if plot_our_code:
        
        for area in last_gen_arena:

            #print(area)

            bl_x = area[0]
            bl_y = frame_height - area[1]

            tr_x = area[2]
            tr_y = frame_height - area[3]


            box_size = tr_x - bl_x

            
            cv2.rectangle(frame, (int(bl_x), int(bl_y)), (int(tr_x), int(tr_y)), green, -1)

            for apple in last_gen_apples[life_steps]:

                if(apple[0] != -1):
                    x_apple = apple[0]
                    y_apple = frame_height - apple[1]


                    # Draw apple position as red circle
                    cv2.circle(frame, (int(x_apple), int(y_apple)), apple_radius, red, -1)  



    x, y = last_generation_positions[life_steps]
    y = frame_height - y
    frame = cv2.circle(frame, (int(x), int(y)), 5, blue, -1)  # Draw agent's position as blue circle



    
    
    # Check if the agent collected an apple
    #for j, area in enumerate(apple_areas):
    #    top_left, box_size = area
    #    if top_left[0] <= x <= top_left[0] + box_size and top_left[1] <= y <= top_left[1] + box_size:
    #        cv2.putText(frame, f'Apple {j+1}', (int(x)-20, int(y)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
    #        cv2.circle(frame, (int(x), int(y)), 10, red, 2)  # Mark apple collection with a red circle

    out.write(frame)  # Write frame to video

out.release()  # Release video writer
