import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def estimated_weights(df):
    '''
    Funtion by Luceri et al. (2020) with slight moderations
    retrieves estimated weights for IRL results
    :param df:
    :return:
    '''
    df_w = pd.DataFrame(columns=["t","p","tw","rt","rp"])
    for i in np.arange(len(df)):
        w_i = df.iloc[i]["w"]
        w_i=w_i[1:]
        w_i = w_i[:-1]
        w_u = np.fromstring(w_i, dtype=float, sep=' ')
        row = [w_u[0],w_u[1],w_u[2],w_u[3],w_u[4]]
        df_w.loc[i]=row
    return df_w


def plot_spiderchart(weights_bots, weights_users):
    mean_bots = weights_bots.mean().tolist()
    mean_humans = weights_users.mean().tolist()
    mean_bots = [*mean_bots, mean_bots[0]]
    mean_humans = [*mean_humans, mean_humans[0]]

    features = ['t', 'p', 'tw', 'rt', 'rp']
    features = [*features, features[0]]

    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(mean_bots))
    plt.figure()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.plot(label_loc, mean_bots, label="Bots", color="firebrick")
    plt.plot(label_loc, mean_humans, label="Humans", color="Navy")
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=features)
    plt.legend(fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.savefig(f'figures/spiderchart_IRL.pdf', dpi=1200, bbox_inches="tight")
    plt.show()


# load data
df_bots = pd.read_csv(f"data/df_results_bots_IRL.csv")
df_users = pd.read_csv(f"data/df_results_users_IRL.csv")

weights_bots = estimated_weights(df_bots)
weights_users = estimated_weights(df_users)

plot_spiderchart(weights_bots, weights_users)
