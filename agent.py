import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import wandb

wandb.init(
    project="snake-ai",

)


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(36, 256,256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_ll = Point(head.x - 40, head.y)
        point_lll = Point(head.x - 60, head.y)
        point_r = Point(head.x + 20, head.y)
        point_rr = Point(head.x + 40, head.y)
        point_rrr = Point(head.x + 60, head.y)
        point_u = Point(head.x, head.y - 20)
        point_uu = Point(head.x, head.y - 40)
        point_uuu = Point(head.x, head.y - 60)
        point_d = Point(head.x, head.y + 20)
        point_dd = Point(head.x, head.y + 40)
        point_ddd = Point(head.x, head.y + 60)

        point_ur = Point(head.x + 20, head.y - 20)
        point_urur = Point(head.x + 40, head.y - 40)
        point_ul = Point(head.x - 20, head.y - 20)
        point_ulul = Point(head.x - 40, head.y - 40)
        point_dr = Point(head.x + 20, head.y + 20)
        point_drdr = Point(head.x + 40, head.y + 40)
        point_dl = Point(head.x - 20, head.y + 20)
        point_dldl = Point(head.x - 40, head.y + 40)

        point_uur = Point(head.x + 20, head.y - 40)
        point_urr = Point(head.x + 40, head.y - 20)
        point_ddr = Point(head.x + 20, head.y + 40)
        point_drr = Point(head.x + 40, head.y + 20)
        point_uul = Point(head.x - 20, head.y - 40)
        point_ull = Point(head.x - 40, head.y - 20)
        point_ddl = Point(head.x - 20, head.y + 40)
        point_dll = Point(head.x - 40, head.y + 20)


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

        state = [
            # Danger up
            (dir_u and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_l)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_r and game.is_collision(point_d)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)),

            # Danger left
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_d)),

            # Danger down
            (dir_u and game.is_collision(point_d)) or
            (dir_r and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_r)),

            # Danger up up
            (dir_u and game.is_collision(point_uu)) or
            (dir_r and game.is_collision(point_rr)) or
            (dir_d and game.is_collision(point_dd)) or
            (dir_l and game.is_collision(point_ll)),

            # Danger right right
            (dir_u and game.is_collision(point_rr)) or
            (dir_r and game.is_collision(point_dd)) or
            (dir_d and game.is_collision(point_ll)) or
            (dir_l and game.is_collision(point_uu)),

            # Danger left left
            (dir_u and game.is_collision(point_ll)) or
            (dir_r and game.is_collision(point_uu)) or
            (dir_d and game.is_collision(point_rr)) or
            (dir_l and game.is_collision(point_dd)),

            # Danger down down
            (dir_u and game.is_collision(point_dd)) or
            (dir_r and game.is_collision(point_ll)) or
            (dir_d and game.is_collision(point_uu)) or
            (dir_l and game.is_collision(point_rr)),

            # Danger up up up
            (dir_u and game.is_collision(point_uuu)) or
            (dir_r and game.is_collision(point_rrr)) or
            (dir_d and game.is_collision(point_ddd)) or
            (dir_l and game.is_collision(point_lll)),

            # Danger right right right
            (dir_u and game.is_collision(point_rrr)) or
            (dir_r and game.is_collision(point_ddd)) or
            (dir_d and game.is_collision(point_lll)) or
            (dir_l and game.is_collision(point_uuu)),

            # Danger left left
            (dir_u and game.is_collision(point_lll)) or
            (dir_r and game.is_collision(point_uuu)) or
            (dir_d and game.is_collision(point_rrr)) or
            (dir_l and game.is_collision(point_ddd)),

            # Danger down down down
            (dir_u and game.is_collision(point_ddd)) or
            (dir_r and game.is_collision(point_lll)) or
            (dir_d and game.is_collision(point_uuu)) or
            (dir_l and game.is_collision(point_rrr)),

            # ------#
            

            # Danger up right
            (dir_u and game.is_collision(point_ur)) or
            (dir_r and game.is_collision(point_dr)) or
            (dir_d and game.is_collision(point_dl)) or
            (dir_l and game.is_collision(point_ul)),

            # Danger up left
            (dir_u and game.is_collision(point_ul)) or
            (dir_l and game.is_collision(point_dl)) or
            (dir_d and game.is_collision(point_dr)) or
            (dir_r and game.is_collision(point_ur)),

            # Danger down right
            (dir_u and game.is_collision(point_dr)) or
            (dir_r and game.is_collision(point_dl)) or
            (dir_d and game.is_collision(point_ul)) or
            (dir_l and game.is_collision(point_ur)),

            # Danger down left
            (dir_u and game.is_collision(point_dl)) or
            (dir_r and game.is_collision(point_ul)) or
            (dir_d and game.is_collision(point_ur)) or
            (dir_l and game.is_collision(point_dr)),

            # Danger up right up right
            (dir_u and game.is_collision(point_urur)) or
            (dir_r and game.is_collision(point_drdr)) or
            (dir_d and game.is_collision(point_dldl)) or
            (dir_l and game.is_collision(point_ulul)),

            # Danger up left up left 
            (dir_u and game.is_collision(point_ulul)) or
            (dir_l and game.is_collision(point_dldl)) or
            (dir_d and game.is_collision(point_drdr)) or
            (dir_r and game.is_collision(point_urur)),

            # Danger down right down right
            (dir_u and game.is_collision(point_drdr)) or
            (dir_r and game.is_collision(point_dldl)) or
            (dir_d and game.is_collision(point_ulul)) or
            (dir_l and game.is_collision(point_urur)),

            # Danger down left down left
            (dir_u and game.is_collision(point_dldl)) or
            (dir_r and game.is_collision(point_ulul)) or
            (dir_d and game.is_collision(point_urur)) or
            (dir_l and game.is_collision(point_drdr)),

            # ------#


            # Danger up up right
            (dir_u and game.is_collision(point_uur)) or
            (dir_r and game.is_collision(point_drr)) or
            (dir_d and game.is_collision(point_ddl)) or
            (dir_l and game.is_collision(point_ull)),

            # Danger up right right
            (dir_u and game.is_collision(point_urr)) or
            (dir_r and game.is_collision(point_ddr)) or
            (dir_d and game.is_collision(point_dll)) or
            (dir_l and game.is_collision(point_ddl)),

            # Danger down right right
            (dir_u and game.is_collision(point_drr)) or
            (dir_r and game.is_collision(point_ddl)) or
            (dir_d and game.is_collision(point_uul)) or
            (dir_l and game.is_collision(point_drr)),

            # Danger down down right
            (dir_u and game.is_collision(point_ddr)) or
            (dir_r and game.is_collision(point_dll)) or
            (dir_d and game.is_collision(point_uur)) or
            (dir_l and game.is_collision(point_urr)),

            # Danger up up left
            (dir_u and game.is_collision(point_uul)) or
            (dir_r and game.is_collision(point_urr)) or
            (dir_d and game.is_collision(point_ddr)) or
            (dir_l and game.is_collision(point_dll)),

            # Danger up left left
            (dir_u and game.is_collision(point_ull)) or
            (dir_r and game.is_collision(point_uur)) or
            (dir_d and game.is_collision(point_drr)) or
            (dir_l and game.is_collision(point_ddl)),

            # Danger down left left
            (dir_u and game.is_collision(point_dll)) or
            (dir_r and game.is_collision(point_uul)) or
            (dir_d and game.is_collision(point_urr)) or
            (dir_l and game.is_collision(point_ddr)),

            # Danger down down left
            (dir_u and game.is_collision(point_ddl)) or
            (dir_r and game.is_collision(point_ull)) or
            (dir_d and game.is_collision(point_uur)) or
            (dir_l and game.is_collision(point_drr)),




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
            # plot(plot_scores, plot_mean_scores)

            plotScore = score
            plotMean = mean_score

            wandb.log({"score": plotScore, "mean_score": plotMean})


if __name__ == '__main__':
    train()