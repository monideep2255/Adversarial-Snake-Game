import pygame
import numpy as np
import time
import random
import copy
import cv2
import numpy as np
import mediapipe as mp
from pygame.constants import K_LEFT
import tensorflow as tf
from tensorflow.keras.models import load_model
from snakegame import *
import multiprocessing
import pyautogui
import util

# pygame 1.9.4
# Hello from the pygame community. https: // www.pygame.org/contribute.html

'''
SnakeGame class represents the game class with both the inputs- Hand recognition input and AI inputs. Everything right now has static inputs.
'''


class SnakeGame:

    def __init__(self, agent) -> None:
        self.display_width = 500
        self.display_height = 500
        self.green = (0, 255, 0)
        self.red = (255, 0, 0)
        self.black = (0, 0, 0)
        self.window_color = (200, 200, 200)
        self.apple = pygame.image.load('apple.jpg')
        self.clock = pygame.time.Clock()
        self.snake_head = [250, 250]
        self.snake_position = [[250, 250], [240, 250], [230, 250]]
        self.apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.score = 0
        self.display = None
        self.direction = 'right'
        
        self.events = []
        self.eventIndex = 0
        if agent == "human":
            self.isPlayedByAI = False
        elif agent == "dfs":
            self.agent = "dfs"
            self.isPlayedByAI = True
        elif agent == "bfs":
            self.agent = "bfs"
            self.isPlayedByAI = True
        else:
            self.agent = "astar"
            self.isPlayedByAI = True

    # def read_inputs(self):

    def collision_with_apple(self, apple_position, score):
        apple_position = [random.randrange(1, 50) * 10, random.randrange(1, 50) * 10]
        self.apple_position = apple_position
        score += 1
        if self.isPlayedByAI:
            self.events = self.event_parser()
            
        return apple_position, score

    def collision_with_boundaries(self, snake_head):
        if snake_head[0] >= 500 or snake_head[0] < 0 or snake_head[1] >= 500 or snake_head[1] < 0:
            return 1
        else:
            return 0

    def collision_with_self(self, snake_position):
        snake_head = snake_position[0]
        if snake_head in snake_position[1:]:
            return 1
        else:
            return 0

    def is_direction_blocked(self, snake_position, current_direction_vector):
        next_step = snake_position[0] + current_direction_vector
        snake_head = snake_position[0]
        if self.collision_with_boundaries(snake_head) == 1 or self.collision_with_self(snake_position) == 1:
            return 1
        else:
            return 0

    def generate_snake(self, snake_head, snake_position, apple_position, button_direction, score):

        if button_direction == 1:
            snake_head[0] += 10
        elif button_direction == 0:
            snake_head[0] -= 10
        elif button_direction == 2:
            snake_head[1] += 10
        elif button_direction == 3:
            snake_head[1] -= 10
        else:
            pass

        if snake_head == apple_position:
            self.apple_position, self.score = self.collision_with_apple(self.apple_position, self.score)
            snake_position.insert(0, list(snake_head))

        else:
            snake_position.insert(0, list(snake_head))
            snake_position.pop()

        return snake_position, self.apple_position, self.score

    def display_snake(self, snake_position):
        for position in snake_position:
            pygame.draw.rect(self.display, self.red, pygame.Rect(
                position[0], position[1], 10, 10))

    def display_apple(self, display, apple_position, apple):
        display.blit(apple, (apple_position[0], apple_position[1]))

    # The main method of the game
    def play_game(self, user_difficulty):

        if self.isPlayedByAI:
            self.events = self.event_parser()
            

        else:
            # Hand gesture code
            mpHands = mp.solutions.hands
            hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            mpDraw = mp.solutions.drawing_utils
            # Load the gesture recognizer model
            model = load_model('mp_hand_gesture')
            # Load class names
            f = open('gesture.names', 'r')
            classNames = f.read().split('\n')
            f.close()
            #Change the number to 0 if you don't have additional webcams
            cap = cv2.VideoCapture(1)

        # Game code
        crashed = False
        prev_button_direction = 1
        button_direction = 1
        if (button_direction == 1) :
            self.direction = 'right'
        if (button_direction == 0) :
            self.direction = 'left'
        if (button_direction == 2) :
            self.direction = 'down'
        if (button_direction == 3) :
            self.direction = 'up'
        current_direction_vector = np.array(
            self.snake_position[0])-np.array(self.snake_position[1])

        i = 0
        while crashed is not True:
            if self.isPlayedByAI:
                if(self.events and self.eventIndex < len(self.events)):
                    pygame.event.post(self.events[self.eventIndex])
                self.eventIndex += 1
            else:

                # hand gesture recognition code
                _, frame = cap.read()
                x, y, c = frame.shape
                frame = cv2.flip(frame, 1)
                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(framergb)
                className = ''

                # post process the result
                if result.multi_hand_landmarks:
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)

                            landmarks.append([lmx, lmy])

                        # Drawing landmarks on frames
                        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                        # Predict gesture
                        prediction = model.predict([landmarks])

                        classID = np.argmax(prediction)
                        
                        className = classNames[classID]
                        # ADDED IN
                # show the prediction on the frame
                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow("Output", frame)
                cv2.moveWindow("Output", 40, 30)

                if (className == "okay") and prev_button_direction != 1:
                    
                    newevent = pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_LEFT, mod=pygame.KMOD_NONE) #create the event
                    pygame.event.post(newevent)

                elif (className == "peace_right") and prev_button_direction != 0:
                    newevent = pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_RIGHT,
                                                  mod=pygame.KMOD_NONE)  # create the event
                    pygame.event.post(newevent)
                elif (className == "thumbs up") and prev_button_direction != 2:
                    newevent = pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_UP,
                                                  mod=pygame.KMOD_NONE)  # create the event
                    pygame.event.post(newevent)
                elif (className == "thumbs down") and prev_button_direction != 3:
                    newevent = pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_DOWN,
                                                  mod=pygame.KMOD_NONE)  # create the event
                    pygame.event.post(newevent)

                if cv2.waitKey(1) == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()

            for event in pygame.event.get():

                if event.type == pygame.KEYDOWN:
                    if (event.key == pygame.K_LEFT) and prev_button_direction != 1:
                        button_direction = 0
                    elif (event.key == pygame.K_RIGHT) and prev_button_direction != 0:
                        button_direction = 1
                    elif (event.key == pygame.K_UP) and prev_button_direction != 2:
                        button_direction = 3
                    elif (event.key == pygame.K_DOWN) and prev_button_direction != 3:
                        button_direction = 2
                    elif event.key == pygame.K_q:
                        crashed = True
                    else:
                        button_direction = button_direction
                        
            if (button_direction == 1) :
                self.direction = 'right'
            if (button_direction == 0) :
                self.direction = 'left'
            if (button_direction == 2) :
                self.direction = 'down'
            if (button_direction == 3) :
                self.direction = 'up'

            self.display.fill(self.window_color)
            self.display_apple(self.display, self.apple_position, self.apple)
            self.display_snake(self.snake_position)

            self.snake_position, self.apple_position, self.score = self.generate_snake(
                self.snake_head, self.snake_position, self.apple_position, button_direction, self.score)

            pygame.display.set_caption("Snake Game" + "  " + "SCORE: " + str(self.score))
            pygame.display.update()
            prev_button_direction = button_direction
            if self.is_direction_blocked(self.snake_position, current_direction_vector) == 1:
                crashed = True

            # NOTE: adjusting clock tick changes the speed of the snake:
            # level 1 = 5
            # level 2 = 10
            # level 3 = 15?
            if (user_difficulty == "easy"):
                self.clock.tick(5)
            elif (user_difficulty == "medium"):
                self.clock.tick(10)
            elif (user_difficulty == "hard"):
                self.clock.tick(15)
        return self.score


    def display_final_score(self, display_text, final_score):
        largeText = pygame.font.Font('freesansbold.ttf', 35)
        TextSurf = largeText.render(display_text, True, self.black)
        TextRect = TextSurf.get_rect()
        TextRect.center = ((self.display_width / 2), (self.display_height / 2))
        self.display.blit(TextSurf, TextRect)

        game_quit = True
        pygame.display.update()
        while game_quit:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_quit = False

    #method calls appropriate agent based on the user configurations and parses the direction in terms of pygame events
    def event_parser(self):
        directions = []
        if self.agent == "astar":
            directions = self.astart_agent()
        elif self.agent == "dfs":
            directions = self.dfs_agent()
        elif self.agent == "bfs":
            directions = self.bfs_agent()

        events = []
        self.eventIndex = 0
        if directions:
            for dir in directions:
                if dir == "up":
                    events.append(pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_UP, mod=pygame.KMOD_NONE))
                elif dir == "down":
                    events.append(pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_DOWN, mod=pygame.KMOD_NONE))
                elif dir == "right":
                    events.append(pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_RIGHT, mod=pygame.KMOD_NONE))
                elif dir == "left":
                    events.append(pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_LEFT, mod=pygame.KMOD_NONE))

        return events
    def dfs_agent(self) -> list:
        # use self.apple_position for the apple position 
        # use self.snake position for the snake position
        '''
        Method should return the list of directions.
        "left" for Left
        "right" for right
        "up" for up
        and "down" for down
        '''
        print("DFS agent onboard")
        # return ["up", "right", "up", "right", "up", "right", "up", "right"]
        visited = []
        stack = util.Stack()
        directions = ['left', 'right', 'up', 'down']

        start_state = self.snake_head
        stack.push([start_state,[]])
        
        if self.apple_position == self.snake_head:
            return []
        
        while not stack.isEmpty():

            currentState = stack.pop()
            currentHeadPosition = currentState[0]
            currentDirections = currentState[1]

            if self.apple_position == currentHeadPosition:
                return currentDirections

            if currentHeadPosition not in visited:
                visited.append(currentHeadPosition)
                for successor in directions:
                    successorCellState = copy.deepcopy(currentHeadPosition)
                    prev_dir = currentDirections[-1] if len(currentDirections) > 0 else self.direction
                    # Remove opposite directions
                    if prev_dir == 'left' and successor == 'right':
                        continue
                    if prev_dir == 'right' and successor == 'left':
                        continue
                    if prev_dir == 'up' and successor == 'down':
                        continue
                    if prev_dir == 'down' and successor == 'up':
                        continue
                    if successor == 'left':
                        successorCellState[0] -= 10
                    elif successor == 'right':
                        successorCellState[0] += 10
                    elif successor == 'up':
                        successorCellState[1] -= 10
                    elif successor == 'down':
                        successorCellState[1] += 10
                    # Remove not possible actions
                    if self.collision_with_self(successorCellState) or self.collision_with_boundaries(successorCellState):
                        
                        continue
                    successorDirections = currentDirections + [successor]
                    if successorCellState not in visited:
                        stack.push([successorCellState, successorDirections])

    def bfs_agent(self):
        print("BFS agent onboard")
        close = []
        open = util.Queue()
        directions = ['left', 'right', 'up', 'down']
        start = copy.deepcopy(self.snake_head)
        open.push([start, []])
        
        if self.apple_position == self.snake_head:
            return []

        while not open.isEmpty():
            currentState = open.pop()
            currentHeadPosition = currentState[0]
            currentDirections = currentState[1]
            if self.apple_position == currentHeadPosition:
                currentDirections.reverse()
                return currentDirections
            if currentHeadPosition not in close:
                close.append(currentHeadPosition)
                for successor in directions:
                    successorCellState = copy.deepcopy(currentHeadPosition)
                    prev_dir = currentDirections[-1] if len(currentDirections) > 0 else self.direction
                    # Remove opposite directions
                    if prev_dir == 'left' and successor == 'right':
                        continue
                    if prev_dir == 'right' and successor == 'left':
                        continue
                    if prev_dir == 'up' and successor == 'down':
                        continue
                    if prev_dir == 'down' and successor == 'up':
                        continue
                    if successor == 'left':
                        successorCellState[0] -= 10
                    elif successor == 'right':
                        successorCellState[0] += 10
                    elif successor == 'up':
                        successorCellState[1] -= 10
                    elif successor == 'down':
                        successorCellState[1] += 10
                    # Remove not possible actions
                    if self.collision_with_self(successorCellState) or self.collision_with_boundaries(successorCellState):
                        
                        continue
                    successorDirections = currentDirections + [successor]
                    if successorCellState not in close:
                        open.push([successorCellState, successorDirections])


    def astart_agent(self):
        print("Astar agent onboard")
        # directions = ["up", "right", "left", "down", "right"]
        
        # using heuristic function that is admissible and consistent
        def manhattan_heuristic(position):
            xy1 = position
            xy2 = self.apple_position
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

        # using the euclidean formula to calculate the cheapest cost of movement
        def euclidean_cost(currentState, nextState): 
            xy1 = currentState
            xy2 = nextState
            return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

        # Step 1: Initialize the start state, the explored nodes array, and the main list to store updates (frontier)
        initialNode = self.snake_head  
        exploredNodes = [] # to store the nodes that have been explored
        frontier = util.PriorityQueue() # to store all the nodes 
        directions = ["left", "right", "up", "down"] # different directions that the snake can move in 

        # Step 2: For A *, the initial node will be pushed on the list
        frontier.push([initialNode,[], 0])
    
        # Step 3: Before proceeding, we shall check if the initial node is the goal node
        if self.apple_position == initialNode:
            return [] 

        print(self.apple_position)

        # Step 4: Main portion of the algorithm: this loop will help us explore the nodes of the graph and find the goal state
        while not frontier.isEmpty():
            currentState = frontier.pop()
            currentHeadPosition = currentState[0]
            currentDirections = currentState[1]
            #currentCost = currentState[2]

            # if goal state is reached, snake grows in length and apple position changes
            if self.apple_position == currentHeadPosition:
                currentDirections.reverse()
                return currentDirections
            if currentHeadPosition not in exploredNodes:
                exploredNodes.append(currentHeadPosition)
                for successor in directions:
                    successorCellState = copy.deepcopy(currentHeadPosition)
                    prev_dir = currentDirections[-1] if len(currentDirections) > 0 else self.direction
                    # Remove opposite directions
                    if prev_dir == 'left' and successor == 'right':
                        continue
                    if prev_dir == 'right' and successor == 'left':
                        continue
                    if prev_dir == 'up' and successor == 'down':
                        continue
                    if prev_dir == 'down' and successor == 'up':
                        continue
                    if successor == 'left':
                        successorCellState[0] -= 10
                    elif successor == 'right':
                        successorCellState[0] += 10
                    elif successor == 'up':
                        successorCellState[1] -= 10
                    elif successor == 'down':
                        successorCellState[1] += 10

                    # Remove not possible actions: as the snake grows, it becomes a threat to itself along with the usual boundaries
                    if self.collision_with_self(successorCellState) or self.collision_with_boundaries(successorCellState):
                        # print(successorCellState, successor)
                        continue
                    successorDirections = currentDirections + [successor]
                    nextCost = euclidean_cost(currentHeadPosition, successorCellState) + manhattan_heuristic(successorCellState)
                    if successorCellState not in exploredNodes:
                        frontier.push([successorCellState, successorDirections, nextCost])

    #main method to start the game
    def startGame(self) -> None:
        # self.detect()
        pygame.init()  # initialize pygame modules

        #### display game window #####

        self.display = pygame.display.set_mode((self.display_width, self.display_height))
        self.display.fill(self.window_color)
        pygame.display.update()

        self.final_score = self.play_game("easy")

        display_text = 'Your Score is: ' + str(self.final_score)
        self.display_final_score(display_text, self.final_score)
        pygame.quit()


if __name__ == "__main__":
    #Change the argument to "human" to play with hand recognition
    game = SnakeGame("dfs")

    game.startGame()
