import matplotlib.pyplot as plt
import numpy as np

def plotSteps(steps):      
    plt.hist(steps, density=True)
    plt.ylabel('')
    plt.xlabel("Steps per round")
    plt.xlim((0,400))
    plt.savefig('stepsHist.png')

steps = np.loadtxt('steps.txt')
plotSteps(steps)


