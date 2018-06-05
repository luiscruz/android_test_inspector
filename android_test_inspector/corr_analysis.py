import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

import pandas
import numpy as np
from biokit.viz import corrplot

FEATURES = [
    'stars',
    'forks',
    'contributors',
    'commits',
    'rating_value',
    'rating_count',
    'sonar_issues_ratio',
    'sonar_blocker_issues_ratio',
    'sonar_critical_issues_ratio',
    'sonar_major_issues_ratio',
    'sonar_minor_issues_ratio',
    "age_numeric",
    "sonar_files_processed",
    "ci/cd",
    "tests",
]

def correlation_matrix_II(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(method="spearman"), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    ax1.set_title('Abalone Feature Correlation')
    labels=FEATURES
    ax1.set_xticks(range(len(labels))-0.5)
    ax1.set_xticklabels(labels,fontsize=6)
    ax1.set_yticks(range(len(labels))-0.5)
    ax1.set_yticklabels(labels,fontsize=6)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()

def fancy_corr_table(df: pandas.DataFrame, output_file, features=FEATURES):
    df[FEATURES].corr().to_csv("./tmp_input.csv", index=False)
    from latex_correlations_matrix import generate_latex_corr_table
    table = generate_latex_corr_table("./tmp_input.csv")
    with open(output_file, 'w') as f:
        f.write(table)

def correlation_matrix(df, features=FEATURES, output_file=None):
    corr = corrplot.Corrplot(df[features].corr(method="spearman"))
    corr.plot(grid=False)
    figure = plt.gcf()
    ax = plt.gca()
    labels = [label.replace("sonar_","").replace("_", " ").title() for label in features]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    if output_file:
        figure.tight_layout()
        figure.savefig(output_file)
    else:
        figure.show()
