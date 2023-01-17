import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss



def estimated_rewards(df):
    """
    Function by Luceri et al. (2020)
    retrieves estimated rewards from the IRL results
    :param df: dataframe of IRL results
    :return: dataframe of only rewards
    """
    df_f = pd.DataFrame(columns=['n+nt','n+rp','n+rt','n+tw','p+nt','p+rp','p+rt','p+tw','t+nt','t+rp','t+rt','t+tw'])
    for i in np.arange(len(df)):
        f_i = df.iloc[i]["r"]
        f_i = f_i[1:]
        f_i = f_i[:-1]
        f_u = np.fromstring(f_i, dtype=float, sep=' ')
        row = [f_u[0],f_u[1],f_u[2],f_u[3],f_u[4],f_u[5],f_u[6],f_u[7],f_u[8],f_u[9],f_u[10],f_u[11]]
        df_f.loc[i]=row
    return df_f

def plot_mean_and_standard_errors(rewards_bots, rewards_users):
    # plots the means and standard errors for each state-action pair
    ticks = [r'$\langle t,tw \rangle$',
             r'$\langle t,rt \rangle$',  # significant
             r'$\langle t,rp \rangle$',
             r'$\langle t,nt \rangle$',
             r'$\langle p,tw \rangle$',
             r'$\langle p,rt \rangle$',
             r'$\langle p,rp \rangle$',
             r'$\langle p,nt \rangle$',  # significant
             r'$\langle n,tw \rangle$',
             r'$\langle n,rt \rangle$',
             r'$\langle n,rp \rangle$',
             r'$\langle n,nt \rangle$']

    data_users = rewards_users.transpose().values.tolist()
    data_bots = rewards_bots.transpose().values.tolist()

    y_bots = [np.mean(x) for x in data_bots]
    y_users = [np.mean(x) for x in data_users]

    err_bots = [ss.sem(x) for x in data_bots]
    err_users = [ss.sem(x) for x in data_users]

    plt.figure()

    plt.errorbar(np.arange(12) * 2.0 - 0.4, y_bots, yerr=err_bots, color='none', ecolor="Firebrick")
    plt.scatter(np.arange(12) * 2.0 - 0.4, y_bots, marker="o", color="Firebrick")
    plt.errorbar(np.arange(12) * 2.0 + 0.4, y_users, yerr=err_users, color='none', ecolor="Navy")
    plt.scatter(np.arange(12) * 2.0 + 0.4, y_users, marker="o", color="Navy")

    plt.plot([], marker='o', c='firebrick', label='Bots')
    plt.plot([], marker='o', c='Navy', label='Humans')

    plt.xticks(np.arange(12) * 2.0, ticks, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(False)

    plt.legend(fontsize=13)
    plt.ylabel('Rewards', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'figures/plot_mean_err_IRL.pdf', dpi=1200, bbox_inches="tight")
    plt.show()


def get_heatmap_values(df_bots, df_users):
    # calculates the means for each state-action pair
    l = []
    for pair in list(df_users.columns):
        state, action = pair.split('+')
        l.append([state,
                  action,
                  np.round(np.mean(df_users[pair]), 3),
                  np.round(np.mean(df_bots[pair]), 3)
                  ])
    df = pd.DataFrame(l, columns=['State', 'Action', 'user', 'bot'])
    return df


def plot_heatmap(df_means, value, min, max):
    pivot = df_means.pivot_table(index="State", columns="Action", values=value)
    sns.set(font_scale=1.3)
    sns.heatmap(pivot, annot=True, vmin=min, vmax=max, cbar_kws={'label': 'Rewards'})
    plt.savefig(f'figures/heatmap_{value}_IRL.pdf', dpi=1200, bbox_inches="tight")


# load data
df_bots = pd.read_csv(f"data/df_results_bots_IRL.csv")
df_users = pd.read_csv(f"data/df_results_users_IRL.csv")

# retrieve estimated rewards
rewards_users = estimated_rewards(df_users)
rewards_users = rewards_users[rewards_users.columns[::-1]]
rewards_bots = estimated_rewards(df_bots)
rewards_bots = rewards_bots[rewards_bots.columns[::-1]]
print(f"There are {len(rewards_users)} users and {len(rewards_bots)} bots.")

# plot means and standard errors for each state-action pair
plot_mean_and_standard_errors(rewards_bots, rewards_users)

# plot heatmap of means for each state and action
heatmap_mean = get_heatmap_values(rewards_bots, rewards_users)
plot_heatmap(heatmap_mean, "bot", "mean", 0, 7)
plot_heatmap(heatmap_mean, "user", "mean", 0, 7)

