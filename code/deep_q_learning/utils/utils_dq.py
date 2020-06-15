from utils.py_torch_agent import *
## Useful functions
dict_conversion = {'identity' : 1,'hot_encoding' : CATALOGUE_SIZE , 'u' : CATALOGUE_SIZE ,\
                   'u_hot' : CATALOGUE_SIZE, 'cached' : 1 ,\
                  'rewards' : CATALOGUE_SIZE,'valuable' :CATALOGUE_SIZE }


def plot_reward_loss(reward, loss, run_mean = 10) :


    f, axs = plt.subplots(1,2,figsize=(22,8))

    smoothed_rews = running_mean(reward, run_mean)
    smoothed_loss = running_mean(loss, run_mean)


    axs[0].plot(smoothed_rews)

    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Rewards through the epochs')

    axs[1].plot(smoothed_loss, color = 'r')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss through the epochs')

    plt.show()


def test_agent(param_deep_Q ) :

    agent, reward = deep_q_learning(**param_deep_Q)

    q_table =  agent.evaluate_q_values(np.arange(50))

    print('Q_table estimated')
    plot_q_table(q_table)
    print('Matrix reward')
    plot_q_table(get_matrix_rewards(param_deep_Q['env']))
    r = [x[1] for x in reward]
    print('Running mean of the reward')
    plt.plot(running_mean(r,100))
    plot_reward_loss(reward,agent.all_loss,run_mean=100)
    print('Cached : {}'.format(np.where(param_deep_Q['env'].cost == 0 )))

    return agent,reward

#

def compare_conversion(name, param_deep_Q, epochs = [1, 100,1000 , 10000], linear = True) :
    '''
    Inputs :

    name : The name of the conversion to be done on the states
    param_deep_Q : The hyperparameters for the deep q learning algorithm

    Output : Plot of the q_table before and after epochs and the list of agents

    '''
    
    param_deep_Q['state_dim'] = dict_conversion[name]
    param_deep_Q['name_conversion_state'] = name

    list_agents = []

    q_tables = [   ]

    rewards = []
    all_loss = []

    
    
    for i in epochs :
        param_deep_Q['max_iter'] = i

        if linear :
            param_deep_Q['model'] = LinearModel( param_deep_Q['state_dim'],CATALOGUE_SIZE)
        else :
            param_deep_Q['model'] = Model( param_deep_Q['state_dim'],CATALOGUE_SIZE)


        agent, reward,loss = deep_q_learning(**param_deep_Q)
        list_agents.append(agent)
        all_loss.append(loss)
        q_table = agent.evaluate_q_values(np.arange(50))
        q_tables.append(q_table)
        rewards.append(reward)



    return q_tables,rewards,list_agents, all_loss




def plot_result_deep_q(epochs,name, q_tables, rewards,list_agents,all_loss, param_deep_Q, rm) :
    # Plot the q_table for the different epochs, the reward matrix and the rewards and loss

    n = len(epochs)
    f, axs = plt.subplots(1,n,figsize=(20,10))
    axs = axs.reshape(-1,1)

    for i in range(n) :

        axs[i][0].imshow(q_tables[i])
        axs[i][0].set_xlabel('Actions')
        axs[i][0].set_ylabel('States')
        axs[i][0].set_title('Q_table for {} epochs'.format(epochs[i]))


    plt.suptitle('Comparison of q_tables for conversion : {}'.format(name), size=20)

    plt.show()

    print('Matrix reward')
    plot_q_table(get_matrix_rewards(param_deep_Q['env']))

    plot_reward_loss(rewards[-1],all_loss[-1],run_mean=rm)



def plot_different_loss(all_loss, rewards, names, rm_loss, rm_reward, title ) :

    f, axs = plt.subplots(1,2,figsize=(16,8))
    axs = axs.reshape(-1,1)
#     plt.subplots_adjust(left=0.125, bottom=0, right=1.9, top=1, wspace=0.1, hspace=0.3)

    for i,loss in enumerate(all_loss) :

        reward = rewards[i]

        smoothed_loss = running_mean(loss, rm_loss)
        smoothed_reward = running_mean( reward , rm_reward)

        axs[1][0].plot(smoothed_loss, label=names[i])

        axs[0][0].plot(smoothed_reward, label = names[i])


    axs[1][0].set_xlabel('Number of iteration')
    axs[1][0].set_ylabel('Loss (running mean of size : {})'.format(rm_loss))
    axs[1][0].legend()
    axs[1][0].grid(True)


    axs[0][0].set_xlabel('Epochs')
    axs[0][0].set_ylabel('Rewards (running mean of size : {})'.format(rm_reward))
    axs[0][0].legend()
    axs[0][0].grid(True)

    f.suptitle(title)

    plt.show()




def compare_q_tables(q_tables, names, title) :

    n = len(q_tables)
    f, axs = plt.subplots(1,n,figsize=(20,10))
    axs = axs.reshape(-1,1)

    for i in range(n) :
        axs[i][0].imshow(q_tables[i])
        axs[i][0].set_xlabel('Actions')
        axs[i][0].set_ylabel('States')
        axs[i][0].set_title('Q_table for state representation : {}'.format(names[i]))


    f.suptitle(title)

    plt.show()
