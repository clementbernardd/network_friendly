from utils.py_torch_agent import *
## Useful functions
dict_conversion = {'identity' : 1,'hot_encoding' : CATALOGUE_SIZE , 'u' : CATALOGUE_SIZE ,\
                   'u_hot' : CATALOGUE_SIZE, 'cached' : 1 ,\
                  'rewards' : CATALOGUE_SIZE,'valuable' :CATALOGUE_SIZE , 'multiple' : 3*CATALOGUE_SIZE,\
                  'multiple_valuable' : 4*CATALOGUE_SIZE}
dic_colors = {'hot_encoding' : 'k' , 'u_hot' : 'y' , 'rewards' : 'c', 'valuable' : 'r', 'multiple' : 'g',
             'multiple_valuable' : 'b'}

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
        q_table = agent.evaluate_q_values(np.arange(CATALOGUE_SIZE))
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



def plot_different_loss(all_loss, rewards, names, rm_loss, rm_reward, title, q_learn = None) :

    f, axs = plt.subplots(1,2,figsize=(16,8))
    axs = axs.reshape(-1,1)

    for i,loss in enumerate(all_loss) :

        reward = rewards[i]

        smoothed_loss = running_mean(loss, rm_loss)
        smoothed_reward = running_mean( reward , rm_reward)

        axs[1][0].plot(smoothed_loss, label=names[i], color = dic_colors[names[i]])

        axs[0][0].plot(smoothed_reward, label = names[i], color = dic_colors[names[i]])


    axs[1][0].set_xlabel('Number of iteration')
    axs[1][0].set_ylabel('Loss (running mean of size : {})'.format(rm_loss))
    axs[1][0].legend()
    axs[1][0].grid(True)

    if q_learn is not None :
        axs[0][0].plot(q_learn, label='Q learning', color ='m')

    # axs[0][0].set_ylim([0,20])
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
        axs[i][0].set_title('State : {}'.format(names[i]))


    f.suptitle(title)

    plt.show()



def compare_q_tables_dic(dic,names = ['hot_encoding','u_hot','rewards'] , gamma = 0 , isLinear = True, tranpose = False) :
    n = len(dic[names[0]][3][-1])
    if isLinear :
        title = 'Q tables for Linear model with gamma = {} after {} epochs'.format(gamma, n)
    else :
        title = 'Q tables for Fully connected model with gamma = {} after {} epochs'.format(gamma, n)
    if tranpose :

        compare_q_tables([dic[x][0][-1].detach().numpy().T for x in names],names,title)
    else :
        compare_q_tables([dic[x][0][-1] for x in names],names,title)

def plot_results_loss_rew_dic(dic, names = ['hot_encoding','u_hot','rewards'], rm_rew = 500, rm_loss = 500, gamma = 0 , q_learn = None) :

    all_loss = [dic[x][3][-1] for x in dic]
    all_rewards = [dic[x][1][-1] for x in dic]

    param = {
    'all_loss' : all_loss, 'rewards' : all_rewards, 'names' : names,'rm_loss' : rm_loss,'rm_reward' : rm_rew,\
    'title' : 'Rewards and Loss for Linear model with gamma = ' + str(gamma), 'q_learn' : q_learn }

    plot_different_loss(**param)




def get_result_tables(param_deep_Q,dic = {}, names = ['hot_encoding','u_hot','rewards'], epochs = [1,5000], linear = True) :
    # Get the results after 5000 epochs for the given parameters
    for x in names :
        dic[x] = compare_conversion(x, param_deep_Q, epochs = epochs, linear = linear)


def get_representation(names,env) :
    # Return a matrix : each line correspond to the conversion of the given state
    represented_tables = []
    for x in names :
        q_t = np.zeros(( CATALOGUE_SIZE,dict_conversion[x]))
        convert = convert = ConversionState(env,x).conversion
        for i in range(CATALOGUE_SIZE) :
            q_t[i,:] = convert(i)
        represented_tables.append(q_t)
    return represented_tables



def get_time_update(dic, name, times, param_deep_Q) :
    # Compute the loss, rewards and q table for different update time for the target network

    all_loss, rewards, q_tables , list_agents = [],[],[],[]

    for time in times :
        param_deep_Q['update_target'] = time

        agent, reward,loss = deep_q_learning(**param_deep_Q)
        list_agents.append(agent)
        all_loss.append(loss)
        q_table = agent.evaluate_q_values(np.arange(CATALOGUE_SIZE))
        q_tables.append(q_table)
        rewards.append(reward)

    dic[name] = [q_tables,all_loss,rewards,list_agents]

def plot_reward_loss_update(rewards, all_loss, times,run_mean = 10) :
    all_colors = ['r','g','b','k','y','m']
    f, axs = plt.subplots(1,2,figsize=(22,8))

    colors = all_colors[:len(rewards)]

    for i,loss in enumerate(all_loss) :
        reward = rewards[i]


        smoothed_rews = running_mean(reward, run_mean)
        smoothed_loss = running_mean(loss, run_mean)


        axs[0].plot(smoothed_rews, color = colors[i], label = 'Update : ' + str(times[i]))
        axs[1].plot(smoothed_loss, color = colors[i],label = 'Update : ' + str(times[i]))


    axs[0].legend()
    # axs[0].set_ylim([0,15])
    axs[0].grid(True)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Rewards through the epochs')

    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss through the epochs')


    plt.show()
