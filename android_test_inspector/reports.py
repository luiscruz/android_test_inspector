import time
from datetime import datetime
from os.path import join as path_join

import click
import matplotlib.pyplot as plt
import pandas

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
    
    # --- Number of projects by framework --- #
    columns = unit_test_frameworks+ui_automation_frameworks+cloud_test_services+ci_services
    colors =  (
        ['c'] * len(unit_test_frameworks) 
        + ['g'] * len(ui_automation_frameworks)
        + ['b'] * len(cloud_test_services)
        + ['m'] * len(ci_services) 
    )

    sums = df[columns].sum()
    labels = sums.index
    heights = sums.values
    figure, ax = plt.subplots(1, 1)

    ax.bar(range(len(labels)), heights, 0.5, color=colors)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.set_xticks(range(len(labels)))
    ax.set_title("Number of projects by test framework")
    ax.set_ylabel("Number of projects")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.grid(linestyle='--')

    def draw_range(ax, xmin, xmax, label):
        y=350
        ax.annotate('', xy=(xmin, y), xytext=(xmax, y), xycoords='data', textcoords='data',
                    arrowprops={'arrowstyle': '|-|', 'color':'black', 'linewidth': 0.5})
        xcenter = xmin + (xmax-xmin)/2
        ytext = y + ( ax.get_ylim()[1] - ax.get_ylim()[0] ) / 20
        ax.annotate(label, xy=(xcenter,ytext), ha='center', va='center', fontsize=9)

    draw_range(ax, -0.5, 3.5, "Unit testing")
    draw_range(ax, 3.5, 11.5, "UI automation")
    draw_range(ax, 11.5, 17.5, "Cloud testing")
    draw_range(ax, 17.5, 21.5, "CI/CD")
    
    figure.tight_layout()
    figure.savefig(path_join(results_output, "framework_count.pdf"))
    # --------------------------------------- #


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
