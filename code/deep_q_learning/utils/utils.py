import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm
import copy
import time
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.environment import *
import pickle


def creation_u(n_states) :
    # Creates a matrix u of related content
    u = np.array([ [random.uniform(0,1) for i in range(n_states)] for j in range(n_states)])
    np.fill_diagonal(u, 0)

    return np.array(u)

def creation_caching(n_states, n_cached) :
    # Returns a list which contains the cost to get a content
    # It contains only 0 (cached) or 1 (non-cached)
    cost = np.array([1 for i in range(n_states)])
    index = list(random.sample(range(0, n_states), n_cached))
    for x in index :
        cost[x] = 0
    return np.array(cost)


def create_priori_popularity(n_states,uniform = True) :
    # It creates a list of probability to get the content
    # This is p_0 = [ p_0, ..., p_K-1]
    # The sum should be one
    if uniform :
        result = (np.ones((1,n_states))/n_states)[0]
    else :
        result = [random.random() for i in range(1,n_states)]
        sum_r = np.sum(result)
        result = [i/sum_r for i in result]
    return result


def get_random_state(p0) :
    # Returns a random content among the contents
    state = [i for i in range(len(p0))]
    return np.random.choice(state,1,p0.tolist())[0]

def get_recommended(state,n_recommended,u) :
    # Returns the n_recommended content of the content 'state'
    # For the moment we only return the most related contents
    liste = u[state]
    not_null = len(np.where(liste != 0)[0])
    indx = (-liste).argsort()[:min(not_null, n_recommended)]
    return np.array(indx)

def get_cached(state, cost):
    # Takes as input a state and the cost matrix (where Xi = 0 if cached and 1 if not-cached)
    # Returns a boolean which says whether or not it is cached or not
    return cost[state] == 0


def plot_penalty(all_penalties, all_rewards, max_x = 10000) :
    # It plots the penalties of the q_learning algorithm
    # It corresponds to the error of recommendation
    fig, arr = plt.subplots(2)
    fig.subplots_adjust(hspace=.5)
    fig.set_size_inches(8, 8)
    index_max = min(max_x, len(all_penalties))

    arr[0].plot(np.arange(len(all_penalties))[:index_max], all_penalties[:index_max])
    arr[0].set_xlabel("Epochs")
    arr[0].set_ylabel("Penalties")
    arr[0].set_title("Error of recommendation")

    arr[1].plot(np.arange(len(all_rewards))[:index_max], all_rewards[:index_max], c='r')
    arr[1].set_xlabel("Epochs")
    arr[1].set_ylabel("Rewards")
    arr[1].set_title("Rewards of recommendation")



    plt.show()


def running_mean(x, N):

    mask=np.ones((1,N))/N
    mask=mask[0,:]
    result = np.convolve(x,mask,'same')

    return result

def plot_q_table(q_table) :
    ax = plt.subplot(111)
    im = ax.imshow(q_table)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.ylabel("States")
    plt.xlabel("Actions")
    plt.title("Q table")
    plt.colorbar(im, cax=cax)

    plt.show()


def plot_multiple_matrices(matrices_x, matrices_y,w,h, x_label, y_label,\
                           title ,liste_titles,sizes = (8,8), is_matrice = False) :
    # Plots multiples matrices

    n = len(matrices_x)
    fig, axes = plt.subplots(w,h,figsize=sizes)
    i=0
    for ax in axes :
        for sub_ax in ax :
            if (i<n):
                if (is_matrice) :
                    sub_ax.imshow(matrices_x[i])

                else :
                    sub_ax.plot(matrices_x[i],matrices_y[i])

                sub_ax.set_xlabel(x_label)
                sub_ax.set_ylabel(y_label)
                sub_ax.set_title(liste_titles[i], fontsize = 12)
                i+=1
            else :
                break



    fig.suptitle(title)
    plt.tight_layout(pad=3.0)
    plt.show()






def plot_penalty_gamma(all_penalties, gammas,running,titles, max_x = 10000, avg = True ) :
    # It plots the penalties of the q_learning algorithm
    # It does it for each value of gamma


    n = len(gammas)
    i=0
    index_max = min(max_x, len(all_penalties[0]))
    X = [np.arange(len(all_penalties[0]))[:index_max] for i in range(n)]
    Y = []
    for i in range(n) :

        if avg :
            Y.append(running_mean(all_penalties[i][:index_max], running))
        else :
            Y.append(all_penalties[i][:index_max])


    liste_title = ["Gamma = {:.1f}".format(gammas[i]) for i in range(n) ]
    plot_multiple_matrices(matrices_x= X, matrices_y=Y,w=5,h=2, x_label="Epoch", y_label=titles,\
                           title=titles + " for different values of gamma" ,\
                           liste_titles=liste_title ,sizes = (8,8), is_matrice = False)




def compare_algo(env, q_table, policy, cmap ) :

    final_table = get_final_table(q_table, binary = True)

    to_plot = [policy,final_table, get_matrix_rewards(env),\
              policy + final_table]
    titles = ["Policy with paper algorithm","Policy with RL algorithm", 'Reward matrix','Sum of two policies']
    plot_multiple(to_plot, 2,2, "Actions", "States","", titles,sizes = (8,8), cmap=cmap)
    similarity = np.where(policy + final_table == 2 )[0].shape[0]/ 50
    print("Similarity between Paper algorithm and RL algorithm : {}".format(similarity))




def plot_multiple(matrices,Nr,Nc, x_label, y_label,title, liste_titles,sizes = (12,12), cmap = 'plasma') :


        fig, axs = plt.subplots(Nr, Nc, figsize=sizes)
        fig.suptitle(title)

        images = []
        k = 0
        for i in range(Nr):
            for j in range(Nc):
                if k>= len(matrices) :
                    break
                images.append(axs[i, j].imshow(matrices[k], cmap=cmap))
                axs[i, j].label_outer()
                axs[i,j].set_xlabel(x_label)
                axs[i,j].set_ylabel(y_label)
                axs[i,j].set_title(liste_titles[k], fontsize = 12)
                k+=1

        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.05)
        plt.show()




def plot_multiple_q_table_gamma(all_q_tables, gammas, title = 'Multiple q_table for gamma') :
    # Plots the q_table through the epochs


    liste_title = ['Gamma = {:.2f}'.format(gammas[k]) for k in range(len(gammas))]
    plot_multiple(all_q_tables,Nr=4,Nc=3, x_label= 'Actions',y_label =  'States',title = title,\
     liste_titles = liste_title ,sizes = (12,12))





def get_matrix_rewards(env) :
    q_table = np.zeros((env.n_states,env.n_actions))
    index_cached = np.where(env.get_index_cached() == 0)[0]
    index_recommended = env.get_index_recommendation()
    q_table[:,index_cached] += 1
    for index,x in enumerate(index_recommended) :
        q_table[index][x] +=1

    return q_table

def get_policy(env) :
    # Returns the policy of the env
    reward = get_matrix_rewards(env)
    return get_final_table(reward)


def get_max_q_table(q_table) :
    # Returns a list of indexes of the max of the q_table
    liste_index = []
    for i in range(q_table.shape[0]) :
        maxm = np.max(q_table[i])
        indexes = np.where(q_table[i] == maxm)[0]
        liste_index.append(indexes)

    return liste_index

def get_final_table(q_table, binary = False) :
    # Retuns the final recommendation
    indexes_max = get_max_q_table(q_table)
    final_table = np.zeros(q_table.shape)
    for index,x in enumerate(indexes_max) :
        for y in x:
            if binary :
                final_table[index,y ] = 1
            else :
                final_table[index,y ] = q_table[index,y ]


    return final_table




def loop_gamma(gammas, max_iter_g) :
    # Run the q_learning algorithm for different values of gammas
    gamma_q_tables, gamma_penalties,gamma_rewards = [], [],[]

    env = Environment(n_actions=50,n_states=50,alpha=0.6, to_leave=0.1, n_recommended=20,\
                 n_cached=10,rewards=[10,5,5,-5],SEED=777)
    for gamma in gammas :
        q_table, all_penalties, all_rewards, all_q_table = q_learning(env,alpha = 0.2,gamma = gamma ,\
                                    epsilon = 0.1,max_iter = max_iter_g)
        gamma_q_tables.append(q_table)
        gamma_penalties.append(all_penalties)
        gamma_rewards.append(all_rewards)

    return gamma_q_tables, gamma_penalties, gamma_rewards



def get_q_table_gamma(gamma_q_tables, gammas) :
    liste_title = ['Gamma = {:.2f}'.format(i) for i in gammas]
    plot_multiple(gamma_q_tables,4,3, "Actions", "States",\
                           "Q table for different values of gamma" ,\
                       liste_titles= liste_title,sizes = (10,10))




def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

def plot_reward(all_reward) :
    # Plot the reward and the running

    eps, rews = np.array(all_reward).T


    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()



def compare_q_tables(q_table, policy,reward_matrix,u , cmap, name=None ) :

    to_plot = [q_table ,  policy,reward_matrix,u ]
    titles = ["Q table","Policy", 'Reward matrix','U matrix']


    f, axs = plt.subplots(2,2,figsize=(20,10))
    axs = axs.reshape(-1,1)

    for i in range(4) :

        axs[i][0].imshow(to_plot[i])
        axs[i][0].set_xlabel('Actions')
        axs[i][0].set_ylabel('States')
        axs[i][0].set_title(titles[i])

    plt.show()


def get_u(env) :
    u = np.zeros((env.n_states, env.n_actions))
    related = env.get_index_recommendation()
    for index, x in enumerate(related) :
        u[index,x] = 1
    return u



def plot_reward_loss(reward, loss, run_mean = 10) :

    eps, rews = np.array(reward).T

    f, axs = plt.subplots(1,2,figsize=(22,8))

    smoothed_rews = running_mean(rews, run_mean)
    smoothed_loss = running_mean(loss, run_mean)


    axs[0].plot(eps[-len(smoothed_rews):], smoothed_rews)
    # axs[0].plot(eps, rews, color='grey', alpha=0.3)
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Total Reward')
    axs[0].set_title('Rewards through the epochs')

    # axs[1].plot(smoothed_loss, color = 'r')
    axs[1].plot(loss, color='grey', alpha=0.3)
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Loss through the epochs')

    plt.show()
