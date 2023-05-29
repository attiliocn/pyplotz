import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

def plot_training_set(ax,x,y, highlights=None):
    if highlights:
        colors = ['gold' if i in highlights else '#0d2d40' for i in x.index]
    else:
        colors = ['#0d2d40' for i in range(len(x))]
    sns.scatterplot(
        ax=ax,
        x=x,
        y=y,
        label='Train',
        s=40,
        alpha=0.80,
        c=colors,
        edgecolor='black',
    )

def plot_test_set(ax,x,y,highlights=None):
    if highlights:
        colors = ['gold' if i in highlights else '#58a3d9' for i in x.index]
    else:
        colors = ['#58a3d9' for i in range(len(x))]
    sns.scatterplot(
        ax=ax,
        x=x,
        y=y,
        label='Test',
        s=55,
        alpha=0.80,
        c=colors,
        edgecolor='black',
        marker='^'
    )

def plot_leave_one_out(ax,x,y):
    sns.scatterplot(
        ax=ax,
        x=x,
        y=y,
        label='LOO',
        s=40,
        alpha=0.85,
        color='none',
        edgecolor='black',
    )

def add_regression_line(ax,x,y):
    sns.lineplot(
        ax=ax,
        x=x,
        y=y,
        linewidth=0.85,
        color='tab:red',
        linestyle='-'
    )

def add_identity_line(ax):
    identity_line = np.arange(*ax.get_xlim())
    sns.lineplot(
        ax=ax,
        x=identity_line,
        y=identity_line,
        color='black',
        alpha=0.25,
        ls='--',
        label='$y=x$'
    )

def add_confidence_interval(ax, y, ci_lower, ci_upper):
    ci_dataset = pd.concat([y, ci_lower, ci_upper], axis=1, ignore_index=True)
    ci_dataset.columns = ['y', 'ci_lower', 'ci_upper']
    ci_dataset.sort_values(by='y', inplace=True)

    ax.fill_between(
        ci_dataset['y'],
        ci_dataset['ci_lower'],
        ci_dataset['ci_upper'],
        linewidth=0,
        alpha=0.15,
        color='tab:gray'
    )

def add_plot_legend(ax):
    ax.legend(
        loc='lower right',
        fontsize=10, 
        ncol=3, 
        markerscale=1, 
        facecolor='white', 
        framealpha=0.35, 
        labelspacing=0.1, 
        handletextpad=0.1,
        columnspacing=0.5,
    )

def add_model_statistics(ax,statistics):
    strings_statistics = []
    for k,v in statistics.items():
        strings_statistics.append("{:<15}{:>6.3f}".format(k,v))

    metrics_box = AnchoredText(
        "\n".join(strings_statistics),
        loc='upper left',
        pad=1,
        frameon=False,
        prop={
            'horizontalalignment':'left',
            'fontsize':8,
            'fontfamily':'monospace',
            'bbox':dict(facecolor='white',alpha=0.5)
        }
    )
    ax.add_artist(metrics_box)