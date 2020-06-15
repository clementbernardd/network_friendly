import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque, namedtuple
SEED = 77
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
from utils.environment import *
from utils.utils import *
CATALOGUE_SIZE = 50

''' Dictionnary conversion '''
# Dictionary with the transformation of states and the size of the states

dict_conversion = {'identity' : 1,'hot_encoding' : CATALOGUE_SIZE , 'u' : CATALOGUE_SIZE ,\
                   'u_hot' : CATALOGUE_SIZE, 'cached' : 1 ,\
                  'rewards' : CATALOGUE_SIZE,'valuable' :CATALOGUE_SIZE }

'''Experience'''
# A class to denote the experience which will be stored in the memory
Experience =namedtuple('Experience', ('state','action','reward','next_state','done'))

''' MEMORY  '''
class Memory(object) :
    """ Memory class that stores the experience of the network"""

    def __init__(self, mem_size) :
        # Mem_size is the number of experiences memorized

        self.buffer = deque(maxlen = mem_size)


    def add(self, experience) :
        # Add an element to the memory

        self.buffer.append(experience)

    def sample(self, batch_size) :
        # Sample randomly among the buffer
        if batch_size > self.buffer.maxlen :
            print("Be careful ! We can't sample more than the size of the memory !")
            return None
        elif len(self.buffer) == 0 :
            print('The buffer is empty ')
            return None

        indexes = np.random.choice(np.arange(len(self.buffer)),\
                                  size = batch_size , replace = False)

        return [self.buffer[i] for i in indexes]

    def size(self) :
        return len(self.buffer)



'''Function to pre-train the memory'''

def pre_trained_mem(mem,pre_train_length, env)  :
    # Initialisation of the memory by taking random actions

    for i in range(pre_train_length) :

        state = env.refresh()

        action = random.randrange(0,env.n_actions,1)

        new_state, reward, done = env.step(action,state)

        mem.add(Experience(state,action,reward,new_state,done))


'''Pre-processing of the states'''


class ConversionState(object) :
    """
    Class that is used to convert state into a new representation
    """
    def __init__(self,  env, name_function, CATALOGUE_SIZE  = CATALOGUE_SIZE) :
        """
        Inputs :

        env : The Environment that simulates the user
        name_function : The name of the function that is used to do the conversion
        CATALOGUE_SIZE : The size of the catalogue
        """

        self.CATALOGUE_SIZE = CATALOGUE_SIZE
        self.env = env
        self.valuable_representation = self.valuable_actions(env)
        self.conversion = self.choose_function(name_function)


    def choose_function(self,name_function) :

        if name_function == 'identity' :
            return self.identity
        elif name_function == 'hot_encoding' :
            return self.hot_encoding
        elif name_function == 'u' :
            return self.convert_u
        elif name_function == 'cached' :
            return self.convert_cached
        elif name_function == 'u_hot' :
            return self.convert_u_hot
        elif name_function == 'rewards' :
            return self.convert_reward
        elif name_function == 'valuable' : 
            return self.valuable

        
    def valuable_actions(self,env) : 
        '''
        Input : environment 
        Output : A representation of the states where each action is increased by 1 when this action is a content either : 
        - related
        - cached
        - Lead to a content cached
        '''
        
        
        states = torch.zeros((env.n_states,env.n_actions),dtype = torch.float)
    
        cost = env.cost
    
        for state in range(states.shape[0]) : 
        
            for i,x in enumerate(cost) : 
                
                if x == 0 : 
                
                    states[state,i] +=1
    
        related = env.recommended
    
        for i in range(states.shape[0]) : 
        
            for new_state in related[i] : 
            
                states[i,new_state] += 1
            
                for k in related[new_state] : 
                    if cost[k] == 0 :
                        states[i,new_state] += 1
        
        return states
    
    def valuable(self, state) : 
        
        return self.valuable_representation[state,:].view(1,CATALOGUE_SIZE)
    
    

    def identity(self, state) :
        '''
        Input : index of the state
        Output : The same state but in TENSOR
        '''
        if state < self.CATALOGUE_SIZE :

            return Tensor([state]).view(-1,1)

        else :
            print('ERROR IN CONVERTION STATE : STATE > CATALOGUE SIZE')
            return None

    def hot_encoding(self, state) :
        '''
        Input : index of the state
        Output : One hot encoding of the state in TENSOR FORMAT
        '''

        if state < self.CATALOGUE_SIZE :
            # Do the encoding
            state_hot_encoded = np.zeros((1,self.CATALOGUE_SIZE))
            state_hot_encoded[0,state] = 1

            return torch.from_numpy(state_hot_encoded).float()
        else :
            print('ERROR IN CONVERTION STATE : STATE > CATALOGUE SIZE')
            return None


    def convert_u(self, state) :
        '''
        Input : index of the state
        Output : Tensor of size the CATALOGUE_SIZE and with the n most related u values
        '''

        new_states = torch.zeros((1, CATALOGUE_SIZE), dtype = torch.float)

        for x in self.env.recommended[state] :

            new_states[0,x] = self.env.u[state, x]

        return new_states

    def convert_u_hot(self, state) :
        '''
        Input : index of the state
        Output : Tensor of size the CATALOGUE_SIZE and with 1 for the n most related u values
        '''

        new_states = torch.zeros((1, CATALOGUE_SIZE), dtype = torch.float)

        for x in self.env.recommended[state] :

            new_states[0,x] = 1

        return new_states


    def convert_cached(self, state) :
        '''
        Input : index of the state
        Output : Tensor of size 1 which says wheter the state is cached or not
        '''
        return Tensor([ not self.env.cost[state]  ]).view(-1,1)


    def convert_reward(self,state) :
        '''
        Input : index of the state
        Output : Reward matrix of the given state : +1 when this is related and +1 when this is cached
        '''

        new_states = self.convert_u_hot(state)
        indexes = np.where(self.env.cost == 0)[0]
        for x in indexes :
            new_states[0,x] +=1

        return new_states
    
    
    
    
    
    
    
    
    




''' Environment '''
class Environment(object) :
    # Creates an environmnent
    # For us, this is the behavior of the User
    def __init__(self,n_actions,  n_states, alpha, to_leave, n_recommended, n_cached,\
                rewards,SEED=77, u=None, cost = None) :
        self.SEED = SEED
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        # Alpha is the coefficient in which a user chooses a recommended content
        self.to_leave = to_leave
        # to_leave is the coefficient in which a user decides to quit the process
        #self.state = np.array([i for i in range(n_states)])
        self.n_recommended = n_recommended
        # n_recommend corresponds to the number of content to recommend. HERE, WE DON'T USE IT YET
        self.n_cached = n_cached
        # n_cached correponds to the number of cached content
        #self.index_recommended = create_matrix_u(self.n_states,self.n_actions,self.n_recommended)
        #self.index_cached = create_matrix_u(self.n_states,self.n_actions,self.n_cached)
        self.u = creation_u(self.n_states) if u is None else u
        # U matrix which denotes the similarity score
        self.cost = creation_caching(self.n_states, self.n_cached) if cost is None else cost
        # It denotes the cached content (0 if cached, 1 not-cached)
        self.p0 = create_priori_popularity(self.n_states)
        # The probability to choose content j from the catalogue
        self.rewards = rewards
        # List of rewards like [RC, RnC , nRC, nRnC]
        # RC = Recommended & Cached ; nRnC = non Recommended & non Cached
        self.recommended = self.get_index_recommendation()

    def refresh(self,SEED = None) :
        # Reset the state to a new value
        if SEED is not None :
            self.SEED = SEED
        return random.randrange(0,self.n_states,1)

    def reset(self) :
        # Reset the environment to a new random state.
        #It resets both the recommendation and the cached indexes
        # It returns a first state
        self.u = creation_u(self.n_states)
        self.cached = creation_caching(self.n_states, self.n_cached)
        self.p0 = create_priori_popularity(self.n_states)

        return self.state[random.randrange(0,self.n_states,1)]



    def find_reward(self, action,state) :
        # Finds the reward for the given action starting in the given state
        # It tests whether the state is recommended, cached or not

        # It gives from the u matrix whether the action from the state is recommended or not
        if ((state is None) or (action is None) ):
            return 0

        recommended_matrix = self.recommended[state]
        is_cached = get_cached(action, self.cost)

        if ( (action in recommended_matrix) and (is_cached)) :
            # Content recommended and cached : best reward
            return self.rewards[0]
        elif ( (action in recommended_matrix) and ( not is_cached) ) :
            # Content recommended but not cached
            return self.rewards[1]
        elif ( (action not in recommended_matrix ) and (is_cached)) :
            # Content not recommended but cached
            return self.rewards[2]
        else :
            # Content neither recommended nor cached
            return self.rewards[3]


    def step(self, action,state) :
        # Returns the state, reward after taking the action in input
        # done is a boolean to say whether the user quits the game or not
        # We want to return the state where will be the user when we suggest him "action"
        # Knowing he is in the current 'state'

        if (random.uniform(0, 1) < self.to_leave) :
            # The user stops to play
            reward = self.find_reward(action,state)
            new_state,reward, done = None,reward,True
            return new_state,reward, done
        else :

            # Else the user will choose among the contents
            if (random.uniform(0,1)< self.alpha ) :

                # Then the user chooses a content among the recommended contentss
                new_state = action
                reward = self.find_reward(action,state)
            else :
                # The user picks a content randomly in the catalogue

                new_state = get_random_state(self.p0)
                reward = self.find_reward(action,state)


            done = False

        if (action == state ) :
            return self.step(action, state)
        else :
            return new_state, reward, done

    def get_index_recommendation(self) :
        # Returns a matrix with all index of recommended content
        recommended = []
        for state in range(self.n_actions) :
            recommended.append(get_recommended(state,self.n_recommended,self.u))
        return recommended

    def get_index_cached(self) :

        return self.cost




''' Model '''

class Model(nn.Module) :
    ''' One fully connected Neural Network'''

    def __init__(self, state_dim , n_actions) :
        """
        Inputs :
        state_dim : The size of the states
        n_actions : The number of actions in the catalogue (i.e the output of the NN)
        """
        super().__init__()

        # Fully connected layer of size 100
        self.hidden = nn.Linear(state_dim , 100)
        # Fully connected layer for the ouput
        self.output = nn.Linear(100, n_actions)

    def forward(self, x) :
        # Forward action to predict the outputs
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)

        return x




''' Linear Model '''

class LinearModel(nn.Module) :
    ''' Linear Model '''

    def __init__(self, state_dim , n_actions) :
        """
        Inputs :
        state_dim : The size of the states
        n_actions : The number of actions in the catalogue (i.e the output of the NN)
        """
        super().__init__()

        # Fully connected layer for the ouput
        self.output = nn.Linear(state_dim, n_actions)

    def forward(self, x) :
        # Forward action to predict the outputs
        x = self.output(x)
        return x




''' Agent '''


class DQAgent(object) :
    '''
    Agent that uses Deep Neural Network to approximate the Q_matrix

    '''
    def __init__(self, n_actions, state_dim , convert_state, mem_size, gamma, epsilon, lr ,model = None, constraints = False,related = None, optimizer = 'SGD') :
        """
        Inputs :

        n_actions : The number of contents that can be recommended
        state_dim : The dimension of the state after convertion
        convert_state : Given a state i, it converts it to the good format for the NN
        mem_size : The size of the Memory that will store the experience
        gamma : Discounted reward parameter
        epsilon : Epsilon-Greedy which decides whether we explore or exploit
        learning_rate : Learning_rate for the Neural Network
        constraints : Boolean whether there are constraints on the recommendation
        recommended : The related contents (if there are constraints)

        Attributes :

        model : The Q_value approximator which is a Neural Network
        memory : The Memory to store the experiences
        loss : The loss used for the learning. By default MSE
        optimizer : The optimizer used to update the weights
        """


        self.n_actions = n_actions
        self.state_dim = state_dim
        # This should be a function
        self.convert_state = convert_state
        self.memory = Memory(mem_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.model =  model 
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr = lr) if optimizer == 'SGD' else optim.ASGD(self.model.parameters(), lr = lr)
        self.all_loss = []
        self.distribution = np.zeros((n_actions,1))
        self.constraints = constraints
        self.related = related

    def memorize(self,state,action,reward,next_state,done) :
        # Store an experience in the Memory
        self.memory.add(Experience(state,action,reward,next_state,done))

    def act(self, state) :
        '''
        EPSILON-GREEDY Decision : Exploitation or Exploration

        INPUT : STATE which is not converted yet

        OUTPUT : An action which is an Integer


        '''
        if np.random.rand() <= self.epsilon:
            # Exploration
            if self.constraints : 
                # Constraints on the recommendation 
                
                recommended_contents =  self.related[state]
                index_state = random.randrange(0,len(recommended_contents),1)
                action = recommended_contents[index_state]
                
            else : 
                # No constraints on the recommendation
                return random.randrange(self.n_actions)
        else :
            # Exploitation
            with torch.no_grad() :
                act_values = self.model(self.convert_state(state))
            return torch.argmax(act_values).int().item()

    def learn(self, batch_size) :
        '''
        Fit the network with a batch of experiences
        Compute the target from this batch
        The target is computed as follows : r + gamma * max(Q(s',a'))
        '''
        minibatch = self.memory.sample(batch_size)
        # Loop over the batch
        batch_loss = 0
        for exp in minibatch:

            self.distribution[exp.action,0] +=1
            
            target = exp.reward
            if not exp.done:
                target = (exp.reward + self.gamma * torch.max(self.model( self.convert_state(exp.next_state)  )).item())
            # Compute the prediction
            with torch.no_grad() :
                target_f = self.model(self.convert_state(exp.state)  )
            # Replace the prediction by the target
            target_f[0,exp.action] = target
            # Fit the neural network with the new target to update the weights
            # Prediction
            prediction = self.model(self.convert_state(exp.state))
            
            # Compute the loss
            current_loss = self.loss(target_f,prediction )

            current_loss.backward()

            batch_loss += current_loss.item()

            self.optimizer.step()

            self.optimizer.zero_grad()
        self.all_loss.append(batch_loss)
        return batch_loss/batch_size


    def save(self, name) :
        # Save the model in 'name' PATH

        torch.save(self.model.state_dict(), name)

    def load(self, name) :
        # Load a model from the 'name' PATH
        self.model.load_state_dict(torch.load(name))


    def evaluate_q_values(self, states) :
        # Given a list of indexes of states, it computes the q_tables
        n = states.shape[0]
        q_table = torch.zeros(n, CATALOGUE_SIZE , dtype = torch.float )
        for i in range(n) :
            with torch.no_grad() :
                q_table[i,:] = self.model(self.convert_state(states[i]))[0]
        return q_table




    def evaluate_policy(self, states) :
        # Given a list of indexes of states, it computes the policy
        n = states.shape[0]
        indexes_max = torch.argmax(self.evaluate_q_values(states) , dim = 1)
        policy = torch.zeros(n , CATALOGUE_SIZE , dtype = torch.int )
        for i in range(n) :

            policy[i,indexes_max[i].item()] = 1
        return policy


''' Deep Q Learning Algorithm '''

def deep_q_learning(env, state_dim, name_conversion_state, mem_size, gamma,\
                    epsilon, learning_rate,max_iter,batch_size ,name, model = None, constraints = False, optimizer = 'SGD') :
    """
    DEEP Q LEARNING ALGORITHM

    Inputs :

    env : The environment that simulates the user behaviour
    state_dim : The dimension of the state after the using the convert_state function
    name_conversion_state : The name of the function that will do the conversion of the state
    mem_size : The size of the Memory that will store the experience
    gamma : Discounted reward parameter
    epsilon : Epsilon-Greedy which decides whether we explore or exploit
    learning_rate : Learning_rate for the Neural Network
    batch_size : The size of the batch which is used to train the network
    name : The name where will be saved the model
    constraints : Boolean to say whether or not we add constraints on the recommendation

    Returns :

    List of rewards and the agent
    """
    # Initialise the conversion for the states
    convert_state = ConversionState(env, name_conversion_state ).conversion

    # Initialisation of the agent
    agent = DQAgent( env.n_actions, state_dim , convert_state ,mem_size, gamma, epsilon, learning_rate, model, constraints,env.recommended, optimizer)
    # List of all the rewards
    all_reward = []
    all_loss = []
    # Fill the memory with random experiences
    if agent.memory.size() < batch_size :

        pre_trained_mem(agent.memory,batch_size + 1, env)

    for e in range(max_iter) :
        state = env.refresh()
        tot_reward = 0
        done = False
        tot_loss = 0
        
        while not done :
            # Simulation until the user leaves

            action = agent.act(state)
            next_state, reward , done = env.step(state, action)
            # Add the experience in the memory
            agent.memorize(state, action, reward, next_state , done)
            
            state = next_state
            tot_reward +=reward

            if done :
                all_reward.append(tot_reward)
                all_loss.append(tot_loss)
                clear_output(True)
                print("Episode: {}/{}, Reward : {}"
                          .format(e, max_iter, tot_reward))
            
                

            # Train the network

            current_loss = agent.learn(batch_size)
            tot_loss += current_loss
            

#     agent.save(name)

    return agent, all_reward, all_loss
