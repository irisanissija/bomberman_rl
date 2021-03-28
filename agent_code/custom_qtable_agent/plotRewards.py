import matplotlib.pyplot as plt
import numpy as np



def plotRewards(rewards):
    plt.plot(rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Game')
    plt.savefig('rewards.png')

def plotRewardsHist(rewards):
    plt.hist(rewards, density=True)
    plt.ylabel('')
    plt.xlabel('Rewards per game')
    min = np.min(rewards)-100
    max = np.max(rewards)-100
    plt.xlim((min, max))
    plt.savefig('rewardsHist.png')



rewards = np.loadtxt('rewards.txt')
#plotRewards(rewards)
plotRewardsHist(rewards)

