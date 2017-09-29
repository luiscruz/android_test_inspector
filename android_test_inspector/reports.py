import time
from datetime import datetime
from os.path import join as path_join
from math import log, floor

import click
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['mathtext.fontset'] = 'cm'

import matplotlib.pyplot as plt
import pandas
import numpy as np
from tabulate import tabulate
import tabulate as T
from scipy.stats import mannwhitneyu
from scipy.stats import ks_2samp
from scipy.stats import shapiro

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
    'codeship',
    'codefresh',
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
    """Generate reports for ICST2018 paper."""
    now = pandas.Timestamp(datetime.now())
    df = pandas.read_csv(
        path_join(results_input, "results_with_coverage.csv"),
        parse_dates=[0,10]
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
    df['sonar_critical_issues_ratio'] = df['sonar_critical_issues'].divide(df['sonar_files_processed'])
    df['sonar_major_issues_ratio'] = df['sonar_major_issues'].divide(df['sonar_files_processed'])
    df['sonar_minor_issues_ratio'] = df['sonar_minor_issues'].divide(df['sonar_files_processed'])
    df_with_google_data = df[~df["rating_count"].isnull()]
    df_with_tests = df[df['tests']]
    df_without_tests = df[~df['tests']]


    colors_dict = {
        'any': 'C0',
        'unit_test_frameworks': 'C1',
        'ui_automation_frameworks': 'C2',
        'cloud_test_services': 'C3',
        'ci_services': 'C4',
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
    sums = df[columns].sum()
    labels = (label in highlights and "• All "+label or label for label in columns)
    labels = [label.title().replace("_", " ") for label in labels]
    heights = sums.values
    figure, ax = plt.subplots(1, 1)
    ax.bar(
        range(len(labels)),
        heights,
        0.5,
        color=colors,
        edgecolor = 'k',
        linewidth= [column in highlights and 0.9 or 0.0 for column in columns]
    )
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xticks(range(len(labels)))
    ax.tick_params(direction='out', top='off')
    # ax.set_title("Number of projects by test framework")
    ax.set_ylabel("Number of projects")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(linestyle='dotted')

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels(["{:.0%}".format(tick/len(df)) for tick in ax2.get_yticks()])
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_ylabel("Percentage of projects")

    def draw_range(ax, xmin, xmax, label):
        y=400
        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
                    arrowprops={'arrowstyle': '|-|', 'color':'black', 'linewidth': 0.5})
        xcenter = xmin + (xmax-xmin)/2
        ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 22
        ax.annotate(label, xy=(xcenter,ytext), ha='center', va='center', fontsize=9)

    draw_range(ax, 0.5, 5.5, "Unit test")
    draw_range(ax, 5.5, 14.5, "UI automation")
    draw_range(ax, 14.5, 21.5, "Cloud test")
    # draw_range(ax, 21.5, 26.5, "CI/CD")

    figure.tight_layout()
    figure.savefig(path_join(results_output, "framework_count.pdf"))
    # --------------------------------------- #

    # --- Percentage of Android tests over the age of the apps --- #
    def tests_in_projects_by_time_of_creation(df_projects, frameworks, label=None,
                                              title=None,
                                              zorder=1, color=None,
                                              verbose=False):
        portions = []
        n_projects_with_tests_history = []
        total_projects_history = []
        age_max = df_projects['age_numeric'].max()+1
        for age in range(age_max):
            n_projects_with_tests = df_projects[df_projects['age_numeric']==age][frameworks].apply(any, axis=1).sum()
            n_projects_with_tests_history.append(n_projects_with_tests)
            total_projects = len(df_projects[df_projects['age_numeric']==age].index)
            total_projects_history.append(total_projects)
            if total_projects == 0:
                portion = 0
            else:
                portion = n_projects_with_tests/total_projects
            portions.append(portion)
            if verbose:
                print("Age {}:".format(age))
                print("{} out of {} projects ({:.1%}).".format(n_projects_with_tests, total_projects, portion))

        plt.plot(range(age_max), portions, label=label, zorder=zorder)
        plt.scatter(range(age_max), portions, total_projects_history, marker='o', linewidth='1', zorder=zorder)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticks(range(age_max))
        ax.set_yticklabels(["{:.0%}".format(label) for label in ax.get_yticks()])
        ax.set_ylabel("Percentage of projects")
        ax.yaxis.grid(linestyle='dotted', color='gray')
        if label:
            legend = ax.legend(loc='upper center', shadow=False)
        if title:
            ax.set_title(title)

    figure, ax = plt.subplots(1,1)
    tests_in_projects_by_time_of_creation(df, unit_test_frameworks+ui_automation_frameworks+cloud_test_services, label="Any", color=colors_dict['any'], zorder=2)
    tests_in_projects_by_time_of_creation(df, unit_test_frameworks, label="Unit tests", color=colors_dict['unit_test_frameworks'], zorder=3)
    tests_in_projects_by_time_of_creation(df, ui_automation_frameworks, label="UI Automation", color=colors_dict['ui_automation_frameworks'], zorder=4)
    tests_in_projects_by_time_of_creation(df, cloud_test_services, label="Cloud testing", color=colors_dict['cloud_test_services'], zorder=5)
    ax.set_xlabel("Years since creation")
    figure.tight_layout()
    figure.savefig(path_join(results_output, "tests_by_age.pdf"))
    # ------------------------------------------------------------ #

    # --- Percentage of 2+years apps with tests grouped by time since last update --- #
    def tests_in_projects_by_time_of_update(df_projects, frameworks, label=None,
                                              title=None,
                                              verbose=False, zorder=None, color=None):
        portions = []
        n_projects_with_tests_history = []
        total_projects_history = []
        age_max = df_projects['time_since_last_update_numeric'].max()+1
        for age in range(age_max):
            n_projects_with_tests = df_projects[df_projects['time_since_last_update_numeric']==age][frameworks].apply(any, axis=1).sum()
            n_projects_with_tests_history.append(n_projects_with_tests)
            total_projects = len(df_projects[df_projects['time_since_last_update_numeric']==age].index)
            total_projects_history.append(total_projects)
            if total_projects == 0:
                portion = 0
            else:
                portion = n_projects_with_tests/total_projects
            portions.append(portion)
            if verbose:
                print("Age {}:".format(age))
                print("{} out of {} projects ({:.1%}).".format(n_projects_with_tests, total_projects, portion))

        plt.plot(range(age_max), portions, label=label, zorder=zorder)
        plt.scatter(range(age_max), portions, total_projects_history, marker='o', linewidth='1', zorder=zorder)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.set_xticks(range(age_max))
        ax.set_yticklabels(["{:.0%}".format(label) for label in ax.get_yticks()])
        ax.set_ylabel("Percentage of projects")
        ax.yaxis.grid(linestyle='dotted', color='gray')

        if label:
            legend = ax.legend(loc='upper center', shadow=False)
        if title:
            plt.title(title)

    figure, ax = plt.subplots(1,1)
    tests_in_projects_by_time_of_update(df_old, unit_test_frameworks+ui_automation_frameworks+cloud_test_services, label="Any", color=colors_dict['any'], zorder=1)
    tests_in_projects_by_time_of_update(df_old, unit_test_frameworks, label="Unit tests", color=colors_dict['unit_test_frameworks'], zorder=2)
    tests_in_projects_by_time_of_update(df_old, ui_automation_frameworks, label="UI Automation", color=colors_dict['ui_automation_frameworks'], zorder=3)
    tests_in_projects_by_time_of_update(df_old, cloud_test_services, label="Cloud testing", color=colors_dict['cloud_test_services'], zorder=4)
    ax.set_xlabel("Years since last update")
    figure.tight_layout()
    figure.savefig(path_join(results_output, "mature_tests_by_update.pdf"))

    # ------------------------------------------------------------------------------- #

    # --- Descriptive stats for popularity metrics --- #
    dictionary = {
        "count": "$N$",
        "mean": "$\\bar{x}$",
        "std": "$s$",
        "min": "$min$",
        "max": "$max$",
        "rating_value": "Rating"
    }
    stats = df[['stars','forks', 'contributors', 'commits', 'rating_value', 'rating_count', 'downloads']].describe()
    stats = stats.applymap((lambda x: "${:.1f}$".format(float(x)))).astype(str)
    stats[['stars','forks', 'contributors', 'commits', 'rating_count']] = stats[['stars','forks', 'contributors', 'commits', 'rating_count']].applymap((lambda x: "${:.0f}$".format(float(x[1:-1])))).astype(str)
    stats.loc['count']= stats.loc['count'].map((lambda x: "${:.0f}$".format(float(x[1:-1])))).astype(str)

    old_escape_rules = T.LATEX_ESCAPE_RULES
    T.LATEX_ESCAPE_RULES = {'%': '\\%'}
    with open(path_join(results_output, "popularity_metrics_stats.tex"), 'w') as f:
        f.write(tabulate(
            stats,
            headers=[dictionary.get(column, column.title().replace("_", " ")) for column in stats.columns],
            showindex=[dictionary.get(name, name) for name in stats.index],
            tablefmt='latex',
            floatfmt=".1f"
        ))
    T.LATEX_ESCAPE_RULES = old_escape_rules
    # -------------------------------------------------- #

    # --- Histogram for downloads --- #
    downloads_distribution = df_with_google_data.groupby('downloads')['downloads'].count()
    heights = df_with_google_data.groupby('downloads')['downloads'].count().values


    figure, ax = plt.subplots(1,1)
    labels = [
        str(human_format(int(cat.split(' - ')[0].replace(',',''))))
        + " – " +
        str(human_format(int(cat.split(' - ')[1].replace(',',''))))
        for cat in downloads_scale
    ]
    # ax.bar(
    #     range(len(labels)),
    #     heights,
    #     width=0.9,
    #     color=[column == '10,000 - 50,000' and 'C1' or 'C0' for column in downloads_scale],
    # )
    downloads_distribution.plot.bar(
        ax=ax,
        width=0.9,
        fontsize=17,
    )
    ax.set_xticklabels(labels, fontsize=17, rotation='vertical')
    ax.set_xlabel("Downloads", fontsize=18)
    ax.set_ylabel("Number of apps", fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.yaxis.grid(linestyle='dotted', color='gray')

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels(["{:.0%}".format(tick/len(df_with_google_data)) for tick in ax2.get_yticks()], fontsize=17)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_ylabel("Percentage of apps", fontsize=18)


    figure.tight_layout()
    figure.savefig(path_join(results_output, "downloads_hist.pdf"))
    # -------------------------------------------------- #

    # ---------- Hypothesis testing ------------- #
    popularity_metrics = [
        'stars',
        'forks',
        'contributors',
        'commits',
        'rating_value',
        'rating_count',
        # 'downloads'
    ]

    def analyze_populations(a,b, continuous=True):
        mean_difference = np.mean(b) - np.mean(a)
        ks_test, ks_p = ks_2samp(a,b)
        mwu_test, mwu_p = mannwhitneyu(a,b, alternative='two-sided')
        
        return {
            # 'MW': "${:.4f}$".format(mwu_p),
            # 'KS': continuous and "${:.4f}$".format(ks_p) or "n.a.",
            'Test': continuous and "${:,.0f}$".format(ks_test) or "${:,.0f}$".format(mwu_test),
            '$p$-value': continuous and "${:.4f}$".format(ks_p) or "${:.4f}$".format(mwu_p),
            '$\\Delta\\bar{x}$': "${:,.2f}$".format(mean_difference),
        }
    tests = [analyze_populations(df_without_tests[metric].dropna(), df_with_tests[metric].dropna(), False) for metric in popularity_metrics]
    keys = [
        'Test',
        '$\mu$-value',
        'mean difference',
    ]
    old_escape_rules = T.LATEX_ESCAPE_RULES
    T.LATEX_ESCAPE_RULES = {'%': '\\%'}
    with open(path_join(results_output, "popularity_metrics_test.tex"), 'w') as f:
        f.write(tabulate(
            tests,
            headers="keys",
            showindex=[metric.title().replace("_"," ") for metric in popularity_metrics],
            tablefmt='latex',

        ))
    T.LATEX_ESCAPE_RULES = old_escape_rules
    # ------------------------------------------- #

    # ---------- Tests vs Rating with Rating count ------------- #
    x = range(0, 10000 , 100)
    y_with_tests = tuple(df_with_tests[df_with_tests['rating_count']>i]['rating_value'].mean() for i in x)
    y_without_tests = tuple(df_without_tests[df_without_tests['rating_count']>i]['rating_value'].mean() for i in x)

    figure, ax = plt.subplots()
    ax.scatter(x, y_with_tests, marker='.', color='C0', label="With tests", zorder=2)
    ax.plot(x, y_with_tests, alpha=0.5, color='C0', zorder=1)
    ax.scatter(x, y_without_tests, marker='2', color='r', label="Without tests", zorder=2)
    ax.plot(x, y_without_tests, alpha=0.5, color='r', zorder=1)
    ax.legend(loc='upper center')

    ax.set_ylabel("Rating")
    ax.set_xlabel("Rating count >")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    figure.tight_layout()
    figure.savefig(path_join(results_output, "rating_with_lower_limit.pdf"))
    # --------------------------------------------------------- #

    # ------------------ CI/CD platforms hist --------------- #

    figure, ax = plt.subplots()
    namepedia={
        "circleci": "Circle CI",
        "travis": "Travis CI",
    }
    df[['ci/cd']+ci_services].sum().plot.bar(
        fontsize=17, edgecolor = 'k', linewidth = [1]+[0]*len(ci_services)
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(linestyle='dotted', color='gray')
    ax.set_ylabel("Number of apps", fontsize=17)
    ax.set_xticklabels(["All"]+[namepedia.get(key, key.title()) for key in ci_services])
    
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticklabels(["{:.0%}".format(tick/len(df)) for tick in ax2.get_yticks()], fontsize=17)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_ylabel("Percentage of apps", fontsize=17)
    
    for p in ax.patches:
        ax.annotate("{:.0f}".format(p.get_height()), (p.get_x() +p.get_width()/2, p.get_height()+4), ha='center', fontsize=15)
    figure.tight_layout()
    figure.savefig(path_join(results_output, "ci_cd_hist.pdf"))
    # ------------------------------------------------------- #

    # ------------------ Sonar vs tests --------------- #
    features = [
        'sonar_issues_ratio',
        'sonar_critical_issues_ratio',
        'sonar_major_issues_ratio',
        'sonar_minor_issues_ratio'
    ]
    names = ['Any', 'Critical', 'Major', 'Minor']
    options = {
        'sym':       '',
        'meanline':  True,
        'showmeans': True,
        'patch_artist': True,
    }
    
    figure, ax = plt.subplots(1,1)
    boxplot = ax.boxplot(
        [
            df_tmp[feature].dropna().values
            for feature in features
            for df_tmp in (df_with_tests, df_without_tests)
        ],
        labels=(
            'With Tests',
            'Without Tests'
        )*4,
        **options
    )
    
    colors = (
        'C0',
        'darkred'
    )*len(features)
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_edgecolor(color)
        patch.set_facecolor((1,1,1,0.8))
    for cap, whisker, color in zip(boxplot['caps'], boxplot['whiskers'], np.repeat(colors,2)):
        cap.set_color(color)
        whisker.set_color(color)
    
    # Big hack for legend
    h1, = ax.plot([1,1], colors[0])
    h2, = ax.plot([1,1], colors[1])
    ax.legend((h1, h2),('With Tests', 'Without Tests'), facecolor='white')
    h1.set_visible(False)
    h2.set_visible(False)
    # -----

    ax.yaxis.grid(linestyle='dotted', color='gray')
    ax.set_xticklabels(names)
    xticks = np.arange(1.5, 4*2+0.5, 2)
    ax.set_xticks(xticks)
    ax.set_ylabel('Number of issues per file')
    ax.set_xlabel('Severity of issues')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)


    mean_differences = [
        df_without_tests[feature].dropna().mean() - 
        df_with_tests[feature].dropna().mean()
        for feature in features
    ]
    median_differences = [
        df_without_tests[feature].dropna().median() - 
        df_with_tests[feature].dropna().median()
        for feature in features
    ]
    pvalues = [
        ks_2samp(df_without_tests[feature].dropna().values,
        df_with_tests[feature].dropna().values, ).pvalue
        for feature in features
    ]
    bbox_props = dict(boxstyle="round,pad=0.3", fc=(1,1,1,0.8), ec='lightgray', lw=0.5)
    for name, x, mean_difference, median_difference, pvalue in zip(names, xticks, mean_differences, median_differences, pvalues):
        ax.annotate(
            (
                r"$\Delta\bar{{x}} = {:.2f}$".format(mean_difference)+"\n"+
                r"$\Delta Md = {:.2f}$".format(median_difference)+"\n"+
                r"$p = {:.4f}$".format(pvalue)
            ),
            (x,4.2),
            va='top', ha='center',
            fontsize=10,
            bbox=bbox_props
        )
    for patch, pvalue in zip(boxplot['boxes'], np.repeat(pvalues,2)):
        if pvalue < 0.05:
            patch.set_facecolor((1.0,1.0,0.8,0.7))
    
    figure.tight_layout()
    figure.savefig(path_join(results_output, "sonar_vs_tests.pdf"))
    
    table = tabulate(
        [
            (
                df_tmp[feature].dropna().count(),
                df_tmp[feature].dropna().median(),
                df_tmp[feature].dropna().mean(),
                df_tmp[feature].dropna().std(),
                shapiro(df_tmp[feature].dropna())[1],
            )
            for feature in features
            for df_tmp in (df_with_tests, df_without_tests)
        ],
        headers=['$N$', '$Md$', '$\\bar{x}', '$s$', '$X \sim N$'],
        showindex=[
            "{} in {}".format(metric, population)
            for metric in names
            for population in ('$W$', '$WO$')
        ],
        tablefmt='latex',
    )
    old_escape_rules = T.LATEX_ESCAPE_RULES
    T.LATEX_ESCAPE_RULES = {'%': '\\%'}
    with open(path_join(results_output, "sonar_metrics.tex"), 'w') as f:
        f.write(table)
    T.LATEX_ESCAPE_RULES = old_escape_rules
    
    # ------------------------------------------------- #


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
