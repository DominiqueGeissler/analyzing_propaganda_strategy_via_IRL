import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


def get_estimated_rewards(df):
    """
    Function by Luceri et al. (2020) with slight moderations
    retrieves estimated rewards from the IRL results
    :param df: dataframe of IRL results
    :return: dataframe of only rewards
    """
    df_f = pd.DataFrame(columns=['n_nt','n_rp','n_rt','n_tw','p-_nt','p-_rp','p-_rt','p-_tw','p+_nt','p+_rp','p+_rt','p+_tw','t_nt','t_rp','t_rt','t_tw'])
    for i in np.arange(len(df)):
        f_i = df.iloc[i]["r"]
        f_i = f_i[1:]
        f_i = f_i[:-1]
        f_u = np.fromstring(f_i, dtype=float, sep=' ')
        row = [f_u[0],f_u[1],f_u[2],f_u[3],f_u[4],f_u[5],f_u[6],f_u[7],f_u[8],f_u[9],f_u[10],f_u[11],f_u[12],f_u[13],f_u[14],f_u[15]]
        df_f.loc[i]=row
    return df_f

def plot_mean_and_standard_errors(ticks, data_bots, data_users, name, ymax):
    y_bots = [np.mean(x) for x in data_bots]
    y_users = [np.mean(x) for x in data_users]

    err_bots = [ss.sem(x) for x in data_bots]
    err_users = [ss.sem(x) for x in data_users]

    fig = plt.figure()
    ax = plt.axes()
    ax.set_facecolor('white')

    plt.errorbar(np.arange(8)*2-0.3, y_bots, yerr=err_bots, color='none', ecolor="Firebrick")
    plt.scatter(np.arange(8)*2-0.3, y_bots, marker="o", color="Firebrick")
    plt.errorbar(np.arange(8)*2+0.3, y_users, yerr=err_users, color='none',ecolor="Navy")
    plt.scatter(np.arange(8)*2+0.3, y_users, marker="o", color="Navy")

    ymin = -3
    ymax = ymax
    plt.vlines(3, ymin=ymin, ymax=ymax, color="lightgrey")
    plt.vlines(7, ymin=ymin, ymax=ymax, color="lightgrey")
    plt.vlines(11, ymin=ymin, ymax=ymax, color="lightgrey")

    plt.xticks(np.arange(8)*2, ticks, fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(False)

    plt.plot([], marker='o', c='firebrick', label='Bots')
    plt.plot([], marker='o', c='Navy', label='Humans')

    plt.legend(fontsize=13)
    plt.ylabel('Rewards', fontsize=13)
    plt.tight_layout()
    plt.savefig(name, dpi=1200, bbox_inches="tight")
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
    sns.heatmap(pivot, annot=True, vmin=min, vmax=max, cbar_kws={'label': 'Rewards'}, xticklabels=[r'$nt$', r'$rp$', r'$rt$', r'$tw$'], yticklabels=[r'$n$', r'$p^{+}$', r'$p^{-}$', r'$t$'])
    plt.savefig(f'..//figures/heatmap_{value}_IRL.pdf', dpi=1200, bbox_inches="tight")


# load data
df_bots = pd.read_csv(f"..//data/df_results_bots_IRL.csv")
df_users = pd.read_csv(f"..//data/df_results_users_IRL.csv")

# retrieve estimated rewards
rewards_users = get_estimated_rewards(df_users)
rewards_users = rewards_users[rewards_users.columns[::-1]]
rewards_bots = get_estimated_rewards(df_bots)
rewards_bots = rewards_bots[rewards_bots.columns[::-1]]
print(f"There are {len(rewards_users)} users and {len(rewards_bots)} bots.")

# plot means and standard errors for each state-action pair
plot_mean_and_standard_errors(rewards_bots, rewards_users)

# plot error bars fpr states t and nt
ticks_tn = [r'$\langle t,tw \rangle$',
    r'$\langle n,tw \rangle$',
    r'$\langle t,rt \rangle$',
    r'$\langle n,rt \rangle$',
    r'$\langle t,rp \rangle$',
    r'$\langle n,rp \rangle$',
    r'$\langle t,nt \rangle$',
    r'$\langle n,nt \rangle$']

data_users_tn = rewards_users[["t_tw", "n_tw", "t_rt", "n_rt", "t_rp", "n_rp", "t_nt", "n_nt"]].transpose().values.tolist()
data_bots_tn = rewards_bots[["t_tw", "n_tw", "t_rt", "n_rt", "t_rp", "n_rp", "t_nt", "n_nt"]].transpose().values.tolist()

name = f"figures/plot_mean_err_IRL_tn.pdf"
plot_mean_and_standard_errors(ticks_tn, data_bots_tn, data_users_tn, name, 13)

# facet_grid_error_plot(data_bots_tn, data_users_tn)

ticks_p = [ r'$\langle p^{+},tw \rangle$',
 r'$\langle p^{-},tw \rangle$',
 r'$\langle p^{+},rt \rangle$',
 r'$\langle p^{-},rt \rangle$',
 r'$\langle p^{+},rp \rangle$',
 r'$\langle p^{-},rp \rangle$',
 r'$\langle p^{+},nt \rangle$',
 r'$\langle p^{-},nt \rangle$',
]

data_users_p = rewards_users[["p+_tw", "p-_tw", "p+_rt", "p-_rt", "p+_rp", "p-_rp", "p+_nt", "p-_nt"]].transpose().values.tolist()
data_bots_p = rewards_bots[["p+_tw", "p-_tw", "p+_rt", "p-_rt", "p+_rp", "p-_rp", "p+_nt", "p-_nt"]].transpose().values.tolist()

name = f"..//figures/plot_mean_err_IRL_p.pdf"
plot_mean_and_standard_errors(ticks_p, data_bots_p, data_users_p, name, 8)

# plot heatmap of means for each state and action
heatmap_mean = get_heatmap_values(rewards_bots, rewards_users)
plot_heatmap(heatmap_mean, "bot", 0, 8)
plot_heatmap(heatmap_mean, "user", 0, 8)

