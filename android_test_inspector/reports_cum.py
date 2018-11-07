import time
from datetime import datetime
from os.path import join as path_join
from math import log, floor

import click
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas
import numpy as np
from tabulate import tabulate
import tabulate as T
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import shapiro
from scipy.stats import ttest_ind
from scipy.stats import zscore
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import chi2_contingency


ui_automation_frameworks = [
    "androidviewclient",
    'appium',
    'calabash',
    'espresso',
    'monkeyrunner',
    'pythonuiautomator',
    'robotium',
    'uiautomator',
]

cloud_test_services = [
    'projectquantum',
    'qmetry',
    'saucelabs',
    'firebase',
    'perfecto',
    'bitbar',
]

unit_test_frameworks = [
    'junit',
    'androidjunitrunner',
    'roboelectric',
    'robospock',
]

ci_services = [
    'travis',
    'circleci',
    'app_veyor',
    'codeship',
    'codefresh',
    'wercker',
]

downloads_scale = [
 '1 - 5',
 '10 - 50',
 '50 - 100',
 '100 - 500',
 '500 - 1,000',
 '1,000 - 5,000',
 '5,000 - 10,000',
 '10,000 - 50,000',
 '50,000 - 100,000',
 '100,000 - 500,000',
 '500,000 - 1,000,000',
 '1,000,000 - 5,000,000',
 '5,000,000 - 10,000,000',
 '10,000,000 - 50,000,000',
 '50,000,000 - 100,000,000',
 '100,000,000 - 500,000,000',
 '500,000,000 - 1,000,000,000',
 '1,000,000,000 - 5,000,000,000',
 '5,000,000,000 - 10,000,000,000',
]

def human_format(number):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    k = 1000.0
    magnitude = int(floor(log(number, k)))
    return '%.0f%s' % (number / k**magnitude, units[magnitude])

@click.command()
@click.option('-i','--results_input', default=".", type=click.Path(exists=True))
@click.option('-o','--results_output', default="./reports", type=click.Path(exists=True))
def reports(results_input, results_output):
    """Generate reports for EMSE paper."""
    now = pandas.Timestamp(2017, 9, 30, 12)
    df = pandas.read_csv(
        path_join(results_input, "results_with_coverage.csv"),
        parse_dates=[0, 10]
    )
    df_googleplay = pandas.read_csv(
        path_join(results_input, "googleplay.csv"),
        index_col='package'
    )
    df = df.join(df_googleplay, on="app_id")
    df_sonar = pandas.read_csv("results_sonar.csv", index_col='package')
    df_sonar.fillna(0, inplace=True)
    df_sonar = df_sonar.add_prefix('sonar_')
    df = df.join(df_sonar, on="app_id")

    #Feature engineering
    df['tests'] = df[unit_test_frameworks+ui_automation_frameworks+cloud_test_services].any(axis=1)
    df['unit_tests'] = df[unit_test_frameworks].apply(any, axis=1)
    df['ui_tests'] = df[ui_automation_frameworks].apply(any, axis=1)
    df["cloud_tests"] = df[cloud_test_services].apply(any, axis=1)
    df["ci/cd"] = df[ci_services].apply(any, axis=1)
    df['age'] = (now - df['created_at'])
    df['age_numeric'] = (now - df['created_at']).astype('<m8[Y]').astype('int')
    df['time_since_last_update'] = (now - df['last_updated'])
    df['time_since_last_update_numeric'] = df['time_since_last_update'].astype('<m8[Y]').astype('int')
    df_old = df[df['age_numeric']>=2]
    df["downloads"] = df["downloads"].astype("category", categories=downloads_scale, ordered=True)
    df['sonar_issues_ratio'] = df['sonar_issues'].divide(df['sonar_files_processed'])
    df['sonar_blocker_issues_ratio'] = df['sonar_blocker_issues'].divide(df['sonar_files_processed'])
    df['sonar_critical_issues_ratio'] = df['sonar_critical_issues'].divide(df['sonar_files_processed'])
    df['sonar_major_issues_ratio'] = df['sonar_major_issues'].divide(df['sonar_files_processed'])
    df['sonar_minor_issues_ratio'] = df['sonar_minor_issues'].divide(df['sonar_files_processed'])
    df_with_google_data = df[~df["rating_count"].isnull()]
    df_with_tests = df[df['tests']]
    df_without_tests = df[~df['tests']]
    df.to_csv("results_merged.csv")


    colors_dict = {
        'any': 'C0',
        'unit_test_frameworks': 'C1',
        'ui_automation_frameworks': 'C2',
        'cloud_test_services': 'C3',
        'ci_services': 'C4',
    }
    
    marker_dict = {
        'any': 'o',
        'unit_test_frameworks': 'v',
        'ui_automation_frameworks': '*',
        'cloud_test_services': 'H',
        'ci_services': 's',
    }

    # --- Number of projects by framework --- #
    columns = (
        ['tests']
        + ['unit_tests'] + unit_test_frameworks
        + ['ui_tests'] + ui_automation_frameworks
        + ['cloud_tests'] + cloud_test_services
        # + ['ci/cd'] + ci_services
    )
    colors =  (
        [colors_dict['any']] +
        [colors_dict['unit_test_frameworks']] * (len(unit_test_frameworks) + 1)
        + [colors_dict['ui_automation_frameworks']] * (len(ui_automation_frameworks) + 1)
        + [colors_dict['cloud_test_services']] * (len(cloud_test_services) + 1)
        + [colors_dict['ci_services']] * (len(ci_services) + 1)
    )

    highlights = [
        'tests',
        'unit_tests',
        'ui_tests',
        'cloud_tests',
        'ci/cd',
    ]
    

    # --- Percentage of Android tests over the age of the apps (cumulated) --- #
    def tests_in_projects_by_time_of_creation_cumm(df_projects, frameworks,
                                                   title=None, verbose=False, **kwargs):
        project_with_test_per_age = []
        total_projects_per_age = []
        n_projects_with_tests_history = []
        total_projects_history = []
        age_max = df_projects['age_numeric'].max()+1
        for age in range(age_max)[::-1]:
            n_projects_with_tests = df_projects[df_projects['age_numeric']==age][frameworks].apply(any, axis=1).sum()
            n_projects_with_tests_history.append(n_projects_with_tests)
            total_projects = len(df_projects[df_projects['age_numeric']==age].index)
            total_projects_history.append(total_projects)
            project_with_test_per_age.append(n_projects_with_tests)
            total_projects_per_age.append(total_projects)
            if verbose:
                print("Age {}:".format(age))
                print("{} out of {} projects ({:.1%}).".format(n_projects_with_tests, total_projects, portion))
        project_with_test_per_age_cum = [sum(project_with_test_per_age[:index+1]) for index in range(len(project_with_test_per_age))]
        total_projects_per_age_cum = [sum(total_projects_per_age[:index+1]) for index in range(len(total_projects_per_age))]
        portions = []
        for with_tests, total in zip(project_with_test_per_age_cum, total_projects_per_age_cum):
            if total > 0:
                portions.append(with_tests/len(df_projects))
            else:
                portions.append(0)
        plt.plot(range(age_max)[::-1], portions, **kwargs)
        # plt.scatter(range(age_max), portions, total_projects_history, marker='o', linewidth='1', zorder=zorder)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks(range(age_max)[::-1])
        ax.set_yticklabels(["{:.0%}".format(label) for label in ax.get_yticks()])
        ax.set_ylabel("Percentage of projects")
        ax.yaxis.grid(linestyle='dotted', color='gray')
        ax.legend(loc='upper center', shadow=False)
        if title:
            ax.set_title(title)

    figure, ax = plt.subplots(1,1)
    tests_in_projects_by_time_of_creation_cumm(
        df,
        unit_test_frameworks+ui_automation_frameworks+cloud_test_services,
        label="Any", color=colors_dict['any'], zorder=2,
        marker=marker_dict['any'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        unit_test_frameworks,
        label="Unit testing", color=colors_dict['unit_test_frameworks'], zorder=3,
        #linestyle=linestyle_dict['unit_test_frameworks']
        marker=marker_dict['unit_test_frameworks'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        ui_automation_frameworks,
        label="GUI testing", color=colors_dict['ui_automation_frameworks'], zorder=4,
        marker=marker_dict['ui_automation_frameworks'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        cloud_test_services,
        label="Cloud testing", color=colors_dict['cloud_test_services'], zorder=5,
        marker=marker_dict['cloud_test_services'],
    )
    ax.set_xlabel("Years since first commit")
    figure.tight_layout()
    figure.savefig(path_join(results_output, "tests_by_age_cumm.pdf"))
    ax.invert_xaxis()
    figure.savefig(path_join(results_output, "tests_by_age_cumm_i.pdf"))
    # ------------------------------------------------------------ #
    

    # --- Percentage of Android tests over the age of the apps (cumulated) --- #
    def tests_in_projects_by_time_of_creation_cumm(df_projects, frameworks,
                                                   title=None, verbose=False, **kwargs):
        project_with_test_per_age = []
        total_projects_per_age = []
        n_projects_with_tests_history = []
        total_projects_history = []
        age_max = df_projects['age_numeric'].max()+1
        for age in range(age_max):
            n_projects_with_tests = df_projects[df_projects['age_numeric']==age][frameworks].apply(any, axis=1).sum()
            n_projects_with_tests_history.append(n_projects_with_tests)
            total_projects = len(df_projects[df_projects['age_numeric']==age].index)
            total_projects_history.append(total_projects)
            project_with_test_per_age.append(n_projects_with_tests)
            total_projects_per_age.append(total_projects)
            if verbose:
                print("Age {}:".format(age))
                print("{} out of {} projects ({:.1%}).".format(n_projects_with_tests, total_projects, portion))
        project_with_test_per_age_cum = [sum(project_with_test_per_age[:index+1]) for index in range(len(project_with_test_per_age))]
        total_projects_per_age_cum = [sum(total_projects_per_age[:index+1]) for index in range(len(total_projects_per_age))]
        portions = []
        for with_tests, total in zip(project_with_test_per_age_cum, total_projects_per_age_cum):
            if total > 0:
                portions.append(with_tests/len(df_projects))
            else:
                portions.append(0)
        plt.plot(range(age_max), portions, **kwargs)
        # plt.scatter(range(age_max), portions, total_projects_history, marker='o', linewidth='1', zorder=zorder)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks(range(age_max))
        ax.set_yticklabels(["{:.0%}".format(label) for label in ax.get_yticks()])
        ax.set_ylabel("Percentage of projects")
        ax.yaxis.grid(linestyle='dotted', color='gray')
        ax.legend(loc='upper center', shadow=False)
        if title:
            ax.set_title(title)

    figure, ax = plt.subplots(1,1)
    tests_in_projects_by_time_of_creation_cumm(
        df,
        unit_test_frameworks+ui_automation_frameworks+cloud_test_services,
        label="Any", color=colors_dict['any'], zorder=2,
        marker=marker_dict['any'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        unit_test_frameworks,
        label="Unit testing", color=colors_dict['unit_test_frameworks'], zorder=3,
        #linestyle=linestyle_dict['unit_test_frameworks']
        marker=marker_dict['unit_test_frameworks'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        ui_automation_frameworks,
        label="GUI testing", color=colors_dict['ui_automation_frameworks'], zorder=4,
        marker=marker_dict['ui_automation_frameworks'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        cloud_test_services,
        label="Cloud testing", color=colors_dict['cloud_test_services'], zorder=5,
        marker=marker_dict['cloud_test_services'],
    )
    ax.set_xlabel("Years since first commit")
    figure.tight_layout()
    figure.savefig(path_join(results_output, "tests_by_age_cumm_2.pdf"))
    # ------------------------------------------------------------ #

    # --- Percentage of Android tests over the age of the apps (cumulated) --- #
    def tests_in_projects_by_time_of_creation_cumm(df_projects, frameworks,
                                                   title=None, verbose=False, **kwargs):
        project_with_test_per_age = []
        total_projects_per_age = []
        n_projects_with_tests_history = []
        total_projects_history = []
        age_max = df_projects['age_numeric'].max()+1
        for age in range(age_max):
            n_projects_with_tests = df_projects[df_projects['age_numeric']==age][frameworks].apply(any, axis=1).sum()
            n_projects_with_tests_history.append(n_projects_with_tests)
            total_projects = len(df_projects[df_projects['age_numeric']==age].index)
            total_projects_history.append(total_projects)
            project_with_test_per_age.append(n_projects_with_tests)
            total_projects_per_age.append(total_projects)
            if verbose:
                print("Age {}:".format(age))
                print("{} out of {} projects ({:.1%}).".format(n_projects_with_tests, total_projects, portion))
        project_with_test_per_age_cum = [sum(project_with_test_per_age[:index+1]) for index in range(len(project_with_test_per_age))]
        total_projects_per_age_cum = [sum(total_projects_per_age[:index+1]) for index in range(len(total_projects_per_age))]
        portions = []
        for with_tests, total in zip(project_with_test_per_age_cum, total_projects_per_age_cum):
            if total > 0:
                portions.append(with_tests/total)
            else:
                portions.append(0)
        plt.plot(range(age_max), portions, **kwargs)
        # plt.scatter(range(age_max), portions, total_projects_history, marker='o', linewidth='1', zorder=zorder)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks(range(age_max))
        ax.set_yticklabels(["{:.0%}".format(label) for label in ax.get_yticks()])
        ax.set_ylabel("Percentage of projects")
        ax.yaxis.grid(linestyle='dotted', color='gray')
        ax.legend(loc='upper center', shadow=False)
        if title:
            ax.set_title(title)

    figure, ax = plt.subplots(1,1)
    tests_in_projects_by_time_of_creation_cumm(
        df,
        unit_test_frameworks+ui_automation_frameworks+cloud_test_services,
        label="Any", color=colors_dict['any'], zorder=2,
        marker=marker_dict['any'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        unit_test_frameworks,
        label="Unit testing", color=colors_dict['unit_test_frameworks'], zorder=3,
        #linestyle=linestyle_dict['unit_test_frameworks']
        marker=marker_dict['unit_test_frameworks'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        ui_automation_frameworks,
        label="GUI testing", color=colors_dict['ui_automation_frameworks'], zorder=4,
        marker=marker_dict['ui_automation_frameworks'],
    )
    tests_in_projects_by_time_of_creation_cumm(
        df,
        cloud_test_services,
        label="Cloud testing", color=colors_dict['cloud_test_services'], zorder=5,
        marker=marker_dict['cloud_test_services'],
    )
    ax.set_xlabel("Years since first commit")
    figure.tight_layout()
    figure.savefig(path_join(results_output, "tests_by_age_cumm_3.pdf"))
    ax.invert_xaxis()
    figure.savefig(path_join(results_output, "tests_by_age_cumm_3_i.pdf"))
    # ------------------------------------------------------------ #



def exit_gracefully(start_time):
    """Print time spent"""
    exit_time = time.time()
    duration = exit_time - start_time
    click.secho(
        "Reports generated in {:.4f} seconds.".format(duration),
        fg='blue'
    )

if __name__ == '__main__':
    start_time = time.time()
    try:
        reports()
    finally:
        exit_gracefully(start_time)
