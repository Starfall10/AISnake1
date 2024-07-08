import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(36, 256, 128, 128, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)


        # Current direction bool
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Current apple direction
        apple_l = game.food.x < game.head.x
        apple_r = game.food.x > game.head.x
        apple_u = game.food.y < game.head.y
        apple_d = game.food.y > game.head.y

        # Current tail direction bool
        tail_dir_l = game.snake[-1].x > game.snake[-2].x
        tail_dir_r = game.snake[-1].x < game.snake[-2].x
        tail_dir_u = game.snake[-1].y > game.snake[-2].y
        tail_dir_d = game.snake[-1].y < game.snake[-2].y


        isUpApple = False
        isUpSnake = False
        isUpWall = True
        count = 1
        while (isUpApple == False and isUpSnake == False and isUpWall == True) and count < 10:
            snake_u = Point(head.x, head.y - 20 * count)
            if game.food == snake_u:
                isUpApple = True
                isUpWall = False
            elif snake_u in game.snake:
                isUpSnake = True
                isUpWall = False
            count += 1

        isUpRightApple = False
        isUpRightSnake = False
        isUpRightWall = True
        count = 1
        while (isUpRightApple == False and isUpRightSnake == False and isUpRightWall == True) and count < 10:
            snake_ur = Point(head.x + 20 * count, head.y - 20 * count)
            if game.food == snake_ur:
                isUpRightApple = True
                isUpRightWall = False
            elif snake_ur in game.snake:
                isUpRightSnake = True
                isUpRightWall = False
            count += 1
        
        isRightApple = False
        isRightSnake = False
        isRightWall = True
        count = 1
        while (isRightApple == False and isRightSnake == False and isRightWall == True) and count < 10:
            snake_r = Point(head.x + 20 * count, head.y)
            if game.food == snake_r:
                isRightApple = True
                isRightWall = False
            elif snake_r in game.snake:
                isRightSnake = True
                isRightWall = False
            count += 1
        
        isDownRightApple = False
        isDownRightSnake = False
        isDownRightWall = True
        count = 1
        while (isDownRightApple == False and isDownRightSnake == False and isDownRightWall == True) and count < 10:
            snake_dr = Point(head.x + 20 * count, head.y + 20 * count)
            if game.food == snake_dr:
                isDownRightApple = True
                isDownRightWall = False
            elif snake_dr in game.snake:
                isDownRightSnake = True
                isDownRightWall = False
            count += 1
        
        isDownApple = False
        isDownSnake = False
        isDownWall = True
        count = 1
        while (isDownApple == False and isDownSnake == False and isDownWall == True) and count < 10:
            snake_d = Point(head.x, head.y + 20 * count)
            if game.food == snake_d:
                isDownApple = True
                isDownWall = False
            elif snake_d in game.snake:
                isDownSnake = True
                isDownWall = False
            count += 1

        
        isDownLeftApple = False
        isDownLeftSnake = False
        isDownLeftWall = True
        count = 1
        while (isDownLeftApple == False and isDownLeftSnake == False and isDownLeftWall == True) and count < 10:
            snake_dl = Point(head.x - 20 * count, head.y + 20 * count)
            if game.food == snake_dl:
                isDownLeftApple = True
                isDownLeftWall = False
            elif snake_dl in game.snake:
                isDownLeftSnake = True
                isDownLeftWall = False
            count += 1

        isLeftApple = False
        isLeftSnake = False
        isLeftWall = True
        count = 1
        while (isLeftApple == False and isLeftSnake == False and isLeftWall == True) and count < 10:
            snake_l = Point(head.x - 20 * count, head.y)
            if game.food == snake_l:
                isLeftApple = True
                isLeftWall = False
            elif snake_l in game.snake:
                isLeftSnake = True
                isLeftWall = False
            count += 1

        isUpLeftApple = False
        isUpLeftSnake = False
        isUpLeftWall = True
        count = 1
        while (isUpLeftApple == False and isUpLeftSnake == False and isUpLeftWall == True) and count < 10:
            snake_ul = Point(head.x - 20 * count, head.y - 20 * count)
            if game.food == snake_ul:
                isUpLeftApple = True
                isUpLeftWall = False
            elif snake_ul in game.snake:
                isUpLeftSnake = True
                isUpLeftWall = False
            count += 1
        

        
        


        state = [

            isUpApple,
            isUpSnake,
            isUpWall,
            isUpRightApple,
            isUpRightSnake,
            isUpRightWall,
            isRightApple,
            isRightSnake,
            isRightWall,
            isDownRightApple,
            isDownRightSnake,
            isDownRightWall,
            isDownApple,
            isDownSnake,
            isDownWall,
            isDownLeftApple,
            isDownLeftSnake,
            isDownLeftWall,
            isLeftApple,
            isLeftSnake,
            isLeftWall,
            isUpLeftApple,
            isUpLeftSnake,
            isUpLeftWall,


            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # # Apple direction
            apple_l,
            apple_r,
            apple_u,
            apple_d,

            # Tail direction
            tail_dir_l,
            tail_dir_r,
            tail_dir_u,
            tail_dir_d,
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()