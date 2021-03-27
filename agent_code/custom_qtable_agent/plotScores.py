import matplotlib.pyplot as plt
import numpy as np


def plotScores(scores):      
    plt.plot(scores)
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.savefig('scores.png')

def plotScoresHist(scores):      
    plt.hist(scores, density=True)
    plt.ylabel('')
    plt.xlabel('Scores per game')
    plt.xlim((0,10))
    plt.savefig('scoresHist.png')


scores = np.loadtxt('scores.txt')
#plotScores(scores)
plotScoresHist(scores)



