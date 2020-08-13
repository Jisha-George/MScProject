# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import * 
from game import Agent
from game import Directions
from graphicsUtils import keys_waiting
from graphicsUtils import keys_pressed
from graphicsUtils import keys_press

import random,util,math,time


class CDAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    
    WEST_KEY  = 'a'
    EAST_KEY  = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY  = 'q'
    SPACE_KEY = 'p'
    ENTER_KEY = 'e'
    
    def __init__(self,  index = 0, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.QValues = util.Counter() #indexed by state and action
        self.lastMove = Directions.STOP
        self.index = index
        self.keypress = []
        Directions.CD = False

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.QValues[state, action]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        values = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if (values):
            return max(values)
        else:
            return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        legal_actions = self.getLegalActions(state) #all the legal actions

        value = self.getValue(state)
        for action in legal_actions:
            if (value == self.getQValue(state, action)):
                return action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """

        keypress = keys_pressed() + keys_waiting()

        if keypress != []:
            self.keypress = keypress

        if (self.SPACE_KEY in self.keypress or 'space' in self.keypress):          
            Directions.CD = True
            
        if Directions.CD == True:
            action = self.CDStart(state)  
          #  print('-' + action)
        elif Directions.CD == False:
            # Pick Action
            action = self.QStart(state)
                
        return action
        
    def QStart(self,state):
      	
        legalActions = self.getLegalActions(state)
        action = None
         
        if (util.flipCoin(self.epsilon)):
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        pacpos = state.getPacmanPosition()
     #   print(pacpos)
     #   print('QS')
     #   print(action)
        print(legalActions)
        return action
    
    def CDStart(self,state):
    #	print('CDS')
        legal = state.getLegalActions(self.index)
        move = self.getMove(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.lastMove in legal:
                move = self.lastMove
        if (self.STOP_KEY in self.keypress) and Directions.STOP in legal: move = Directions.STOP

        if move not in legal: move = Directions.STOP

        self.lastMove = move
        pacpos = state.getPacmanPosition()
        #print(pacpos)
        return move
    
    def getMove(self, legal):
        move = Directions.STOP
        if   (self.WEST_KEY in self.keypress or 'Left' in self.keypress) and Directions.WEST in legal:  
            move = Directions.WEST
         #   print('1' + move)
          #  print ('Left')
        if   (self.EAST_KEY in self.keypress or 'Right' in self.keypress) and Directions.EAST in legal: 
            move = Directions.EAST
           # print('2' + move)
            #print ('Right')
        if   (self.NORTH_KEY in self.keypress or 'Up' in self.keypress) and Directions.NORTH in legal:   
            move = Directions.NORTH
   #         print('3' + move)
    #        print ('Up')
        if   (self.SOUTH_KEY in self.keypress or 'Down' in self.keypress) and Directions.SOUTH in legal: 
            move = Directions.SOUTH
     #       print('4' + move)
      #      print ('Down')
        if (self.ENTER_KEY in self.keypress or 'Return' in self.keypress): 
            Directions.CD = False
            move = Directions.STOP
       #     print('5' + move)
        #    print('Enter')
        return move
   
    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        newQValue = (1 - self.alpha) * self.getQValue(state, action) #new Qvalue
        newQValue += self.alpha * (reward + (self.discount * self.getValue(nextState)))
        self.QValues[state, action] = newQValue  
        print(self.episodesSoFar)
        print(state)
        print(nextState)
        print('1 ' + action)
#        print(newQValue)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class PacmanCDAgent(CDAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        CDAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = CDAgent.getAction(self,state)
        self.doAction(state,action)
#        time.sleep(1)
        return action


class ApproximateQCDAgent(PacmanCDAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanCDAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        print(self.weights)
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """

        features = self.featExtractor.getFeatures(state,action)
        QValue = 0.0

        for feature in features:
            QValue += self.weights[feature] * features[feature]
            print(QValue) 
        return QValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        QValue = 0
        difference = reward + (self.discount * self.getValue(nextState) - self.getQValue(state, action))
        features = self.featExtractor.getFeatures(state, action)

        for feature in features:
          self.weights[feature] += self.alpha * features[feature] * difference
          

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanCDAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            
            pass
