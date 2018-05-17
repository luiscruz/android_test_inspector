from os.path import join as path_join

import click
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt

import pandas
import numpy as np
from tabulate import tabulate
import tabulate as T

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
    "age",
    "sonar_files_processed",
    "ci/cd",
    'tests',
]

def correlation_matrix(df):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
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

def generate_corr_table(df, output_file, features=FEATURES):
    # print(df[FEATURES].corr())
    # correlation_matrix(df[FEATURES])
    fancy_corr_table(df, output_file)


# old_escape_rules = T.LATEX_ESCAPE_RULES
# T.LATEX_ESCAPE_RULES = {'%': '\\%'}
# table = tabulate(
#     [
#         (
#             sample_name,
#             df_tmp[feature].dropna().count(),
#             "${:.2f}$".format(df_tmp[feature].dropna().median()),
#             "${:.2f}$".format(df_tmp[feature].dropna().mean()),
#             "${:.2f}$".format(df_tmp[feature].dropna().std()),
#             shapiro(df_tmp[feature].dropna())[1] < 0.0001 and "$p < 0.0001$",
#         )
#         for feature in features
#         for (df_tmp, sample_name) in ((df_with_tests, '$W$'), (df_without_tests, '$WO$'))
#     ],
#     headers=['Tests', '$N$', '$Md$', '$\\bar{x}$', '$s$', '$X \sim N$'],
#     showindex=issues_column,
#     tablefmt='latex',
# )
# T.LATEX_ESCAPE_RULES = old_escape_rules
# with open(path_join(results_output, "sonar_metrics.tex"), 'w') as f:
#     f.write(table)
# # ------------------------------------------------- #
