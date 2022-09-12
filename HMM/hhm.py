from hmmlearn import hmm
import numpy as np
import math

states = ('Rainy','Sunny')
observations = ('walk','shop','clean')
start_probability = {"Rainy":0.6, "Sunny":0.4}
transition_probability = {"Rainy": {"Rainy":0.7, "Sunny":0.3},
                          'Sunny': {"Rainy":0.4,'Sunny':0.6}}

emission_probability = {
    'Rainy': {'walk':0.1,'shop':0.4,'clean':0.5},
    'Sunny': {'walk':0.6,'shop':0.3,'clean':0.1},

}

if __name__ == '__main__':
    model = hmm.MultinomialHMM(n_components=2)
    model.startprob_ = np.array([0.6,0.4])
    model.transmat_ = np.array([[0.7,0.3],
                                [0.4,0.6]])
    model.emissionprob_ = np.array([[0.1,0.4,0.5],
                                    [0.6,0.3,0.1]])

    print(math.exp(model.score(np.array([[0, 0, 2]]))))