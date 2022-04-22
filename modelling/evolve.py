import sys
sys.path.append("./")



import os
import numpy as np
from neat import nn, population, statistics, parallel
from modelling.environment import RLEnvironment


"""
adapted from
"""


MAX_STEPS = 1000
EPISODES = 1
RENDER = False
GENERATIONS = 50
NUM_CORES = 4


env = RLEnvironment()


def simulate_species(net, env, episodes=1, steps=5000, render=False):
    fitnesses = []
    for runs in range(episodes):
        inputs = env.reset()
        cum_reward = 0.0
        for j in range(steps):
            outputs = net.serial_activate(inputs)
            # action = np.argmax(outputs)
            inputs, reward, done, _ = env.step(outputs)
            if render:
                env.render()
            if done:
                break
            cum_reward += reward

        fitnesses.append(cum_reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness


def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, env, EPISODES, MAX_STEPS, render=RENDER)


def train_network(env):

    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, env, EPISODES, MAX_STEPS, render=RENDER)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'gym_config')
    pop = population.Population(config_path)
    # Load checkpoint
    # if args.checkpoint:
    #     pop.load_checkpoint(args.checkpoint)
    # Start simulation
    if RENDER or NUM_CORES == 1:
        pop.run(eval_fitness, GENERATIONS)
    else:
        pe = parallel.ParallelEvaluator(NUM_CORES, worker_evaluate_genome)
        pop.run(pe.evaluate, GENERATIONS)

    pop.save_checkpoint("checkpoint")

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    import pickle
    with open('winner.pkl', 'wb') as output:
       pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    # raw_input("Press Enter to run the best genome...")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, env, 1, MAX_STEPS, render=True)



# env = gym.make(game_name)
if __name__ == "__main__":
    train_network(env)
