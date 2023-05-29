import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pygame
import time
import random




#### initialize game #######
block_size = 10
screen_size_x = 80*block_size
screen_size_y = 60*block_size
speed = 100
black = (0,0,0)
white = (255,255,255)
red   = (255, 0, 0)
green = (0,255,0)
blue  = (0, 0, 255)
yellow = (255, 255, 0)
pygame.init()
screen = pygame.display.set_mode((screen_size_x, screen_size_y))
pygame.display.set_caption('snake game by austin 3:16')
game_clock = pygame.time.Clock()
score_increase_per_food = 10
negative_reward_value = -20
positive_reward_value = 10
reward_array = [[0]]
state_action_array = [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]
count = [0]
####################################################



def determine_present_state(screen_size_x, screen_size_y, present_direction, food_position, snake_structure):
    food_left_right = 0         # state 0
    distance_x_to_food = snake_structure[0][0]-food_position[0]
    if ( distance_x_to_food < 0):
        food_left_right = -1
    if ( distance_x_to_food > 0):
        food_left_right = 1

    food_up_down = 0            # state 1
    distance_y_to_food = snake_structure[0][1]-food_position[1]
    if (distance_y_to_food<0):
        food_up_down = -1
    if (distance_y_to_food>0):
        food_up_down = 1

    close_to_left_boundary = 0   #state 2
    distance_to_left_boundary = snake_structure[0][0] - 0
    if (distance_to_left_boundary == block_size):
        close_to_left_boundary = 1

    close_to_down_boundary = 0   #state 3
    distance_to_down_boundary = snake_structure[0][1] - 0
    if (distance_to_down_boundary == block_size):
        close_to_down_boundary = 1

    close_to_right_boundary = 0   #state 4
    distance_to_right_boundary = screen_size_x - snake_structure[0][0]
    if (distance_to_right_boundary == block_size):
        close_to_right_boundary = 1
    
    close_to_up_boundary = 0   #state 5
    distance_to_up_boundary = screen_size_y - snake_structure[0][1]
    if (distance_to_up_boundary == block_size):
        close_to_up_boundary = 1

    up_direction = 0         #state 6
    down_direction = 0       #state 7
    left_direction = 0       #state 8
    right_direction = 0      #state 9
    if present_direction == 'up':
        up_direction = 1
    if present_direction == 'down':
        down_direction = 1
    if present_direction == 'left':
        left_direction = 1
    if present_direction == 'right':
        right_direction = 1
    
    close_to_body_right = 0  #state 10
    close_to_body_left = 0   #state 11
    close_to_body_up = 0     #state 12
    close_to_body_down = 0   #state 13
    temp_1 =[1000]
    temp_2 =[1000]
    temp_3 =[1000]
    temp_4 =[1000]
    for position in snake_structure[1:]:
        temp_x = snake_structure[0][0]-position[0]
        if (temp_x > 0 and (snake_structure[0][1] == position[1])):
            temp_1.append(abs(temp_x))
        if (temp_x <= 0 and (snake_structure[0][1] == position[1])): 
            temp_2.append(abs(temp_x))
        temp_y = snake_structure[0][1]-position[1]
        if (temp_y >0 and (snake_structure[0][0] == position[0])):
            temp_3.append(abs(temp_y))
        if(temp_y <=0 and (snake_structure[0][0] == position[0])):
            temp_4.append(abs(temp_y))
    
    distance_to_body_left = min(temp_1)
    if (distance_to_body_left == block_size):
        close_to_body_left = 1

    distance_to_body_right = min(temp_2)
    if (distance_to_body_right == block_size):
        close_to_body_right = 1

    distance_to_body_up = min(temp_3)
    if (distance_to_body_up == block_size):
        close_to_body_up = 1

    distance_to_body_down = min(temp_4)
    if (distance_to_body_down == block_size):
        close_to_body_down = 1

    return [food_left_right, food_up_down, close_to_left_boundary, close_to_down_boundary, close_to_right_boundary, close_to_up_boundary, up_direction, down_direction, left_direction, right_direction, close_to_body_right, close_to_body_left, close_to_body_up, close_to_body_down]



def one_hot_action(present_direction):
    up_direction = 0
    down_direction = 0
    left_direction = 0
    right_direction = 0
    if present_direction == 'up':
        up_direction = 1
    if present_direction == 'down':
        down_direction = 1
    if present_direction == 'left':
        left_direction = 1
    if present_direction == 'right':
        right_direction = 1
    return [up_direction, down_direction, left_direction, right_direction]



def moved_closer_to_food_reward(old_head_position, new_head_position, food_position):
    reward = 0
    old_distance_to_food = ( (old_head_position[0] - food_position[0] )**2 ) + ( (old_head_position[1] - food_position[1] )**2 )
    new_distance_to_food = ( (new_head_position[0] - food_position[0] )**2 ) + ( (new_head_position[1] - food_position[1] )**2 )
    if (new_distance_to_food < old_distance_to_food):
        reward = positive_reward_value
    return reward



def policy_Random(previous_direction):                  
    array_of_direction = ['up', 'down', 'left', 'right']
    x = random.choice(array_of_direction)

    if x == 'up'and previous_direction == 'down':
        x = 'down'
    if x == 'down'and previous_direction == 'up':
        x = 'up'
    if x == 'left' and previous_direction == 'right':
        x = 'right'
    if x == 'right' and previous_direction == 'left':
        x = 'left'
    new_direction = x
    return new_direction



def training_function(x_train, y_train):
    function = MLPRegressor(solver ='adam', alpha=1e-5, hidden_layer_sizes=(40), random_state=1, max_iter=10000).fit(x_train, y_train)
    return function
    


def policy_RL(current_state, current_direction, trained_nn):
    predicted_reward_dictionary ={}
    array_of_direction = ['up', 'down', 'left', 'right']
    if current_direction == 'up':
        array_of_direction.remove('down')
    if current_direction == 'down':
        array_of_direction.remove('up')
    if current_direction == 'left':
        array_of_direction.remove('right')
    if current_direction == 'right':
        array_of_direction.remove('left')
    for direction in array_of_direction:
        one_hot_snake_direction = one_hot_action(direction)
        state_action_combination = current_state + one_hot_snake_direction
        input_to_nn = np.array(state_action_combination).reshape(1, -1)
        predicted_reward = trained_nn.predict(input_to_nn) 
        predicted_reward_dictionary[direction]= predicted_reward
    optimal_direction = max(predicted_reward_dictionary, key=predicted_reward_dictionary.get)
    return optimal_direction



def new_position_of_snake(previous_position, direction):
    new_position = previous_position
    if direction == 'up':
        new_position[1] = new_position[1]-block_size
    if direction == 'down':
        new_position[1] = new_position[1]+block_size
    if direction == 'left':
        new_position[0] = new_position[0]-block_size
    if direction == 'right':
        new_position[0] = new_position[0]+block_size
    return new_position



def is_fruit_eating(position_of_snake_head, position_of_fruit):
    if position_of_snake_head[0] == position_of_fruit[0] and position_of_snake_head[1] == position_of_fruit[1]:
        decision = True
    else:
        decision = False
    return decision



def grow_snake(snake_structure, snake_head):
    snake_structure.insert(0,list(snake_head))



def draw_snake_and_fruit(snake_structure, fruit_position):
    screen.fill(blue)
    pygame.draw.circle(screen, white, (snake_structure[0][0]+block_size/2, snake_structure[0][1]+block_size/2),block_size/2, block_size)
    for position in snake_structure[1:]:
        pygame.draw.rect(screen, white, pygame.Rect(position[0], position[1], block_size, block_size ))
    pygame.draw.rect(screen, green, pygame.Rect(fruit_position[0], fruit_position[1], block_size, block_size))



def did_snake_head_touch_border(position):
    if position[0]>=screen_size_x or position[0]<=0 or position[1]>=screen_size_y or position[1]<=0:
        return True
    else:
        return False
    


def did_snake_bite_body(snake_head, snake_body):
    if snake_head in snake_body[1:]:
        return True
    else:
        return False



def display_text_on_screen(caption, colour, position_x, position_y, font, size):
    display_font = pygame.font.SysFont(font, size)
    display_surface = display_font.render(caption,True, colour)
    display_rectangle = display_surface.get_rect()
    screen.blit(display_surface, (position_x, position_y))



def display_state_on_screen(present_state, colour, position_x, position_y, font, size):
    for x in range(6,10):
        if abs(present_state[x]) == 1000:
            present_state[x] = 'NA'
    display_text_on_screen("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(int(present_state[0]), int(present_state[1]), int(present_state[2]), int(present_state[3]), int(present_state[4]), int(present_state[5]), (present_state[6]),(present_state[7]), (present_state[8]), (present_state[9]), int(present_state[10]), int(present_state[11]), int(present_state[12]), int(present_state[13]), int(present_state[14]), int(present_state[15]), int(present_state[16]), int(present_state[17])), colour, position_x, position_y, font, size)
    display_text_on_screen("x_food, y_food, <-edge, v-edge, ->edge, ^-edge, (Direction) up, dn, lf, rt, ->body, <-body, ^-body, v-body, (Action) up, dn, lf, rt", colour, position_x, position_y + 1.5*size, font, 15)



def should_game_quite():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()



def to_continue_game(score):
    screen.fill(blue)
    display_text_on_screen("GAME OVER. Your score is {}. wait a moment for game to restart, snake will learn from experience ".format(score), yellow, 0 , screen_size_y/2, "times new roman", 20)
    pygame.display.update()
    game_clock.tick(1)   



def snake_game_loop():
    count[0] += 1
    trained_nn = training_function(state_action_array, reward_array)
    snake_head_position = [screen_size_x/2, screen_size_y/2]
    snake_structure = []
    snake_structure.append(snake_head_position)
    snake_direction = 'up'
    food_position = [block_size*random.randrange(1, screen_size_x/block_size - 1), block_size*random.randrange(1, screen_size_y/block_size - 1) ]

    score = 0
    game_on = True
    while game_on:
        old_head_position = snake_structure[0]
        present_state = determine_present_state(screen_size_x, screen_size_y, snake_direction, food_position, snake_structure)
        draw_snake_and_fruit(snake_structure, food_position)
        if (count[0]%5 != 4 and count[0]<10):
            snake_direction = policy_Random(snake_direction)
            display_text_on_screen(" random policy to aid learning ", yellow, screen_size_x/2,0, "times new roman", 25)
        else:
            snake_direction = policy_RL(present_state, snake_direction, trained_nn)
            display_text_on_screen("Reinforcement learning based action", yellow, screen_size_x/2,0, "times new roman", 25)

        one_hot_snake_direction = one_hot_action(snake_direction)
        state_action_combination = present_state + one_hot_snake_direction
        snake_head_position = new_position_of_snake(snake_head_position, snake_direction)
        grow_snake(snake_structure, snake_head_position)
        
        reward = moved_closer_to_food_reward(old_head_position, snake_structure[0], food_position)
        
        if snake_head_position == food_position:
            score += score_increase_per_food
            food_position = [block_size*random.randrange(1, screen_size_x/block_size - 1), block_size*random.randrange(1, screen_size_y/block_size - 1) ]
        else:
            del(snake_structure[-1])

        check1 = did_snake_head_touch_border(snake_head_position)
        check2 = did_snake_bite_body(snake_head_position, snake_structure)
        if (check1 == True or check2 == True):
            reward += negative_reward_value

        if (state_action_combination not in state_action_array ):
            state_action_array.append(state_action_combination)
            reward_array.append([reward])

        if (check1 == True or check2 == True):
            to_continue_game(score)
            snake_game_loop()

        #display_text_on_screen(" reward is {}".format(reward), yellow, screen_size_x/4,0, "times new roman", 25)
        #display_state_on_screen(state_action_combination, yellow, 0, screen_size_y/2, "times new roman", 20)
        display_text_on_screen(" score is {}".format(score), yellow, 0,0, "times new roman", 25)
        pygame.display.update()
        should_game_quite()
        game_clock.tick(speed)




    pygame.quit()
    quit()


snake_game_loop()
    

















