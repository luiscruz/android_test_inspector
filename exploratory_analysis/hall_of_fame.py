import pandas
from datetime import datetime

now = pandas.Timestamp(datetime.now())
df_simple = pandas.read_csv("../results_with_coverage.csv", parse_dates=[0,10])
df_googleplay = pandas.read_csv("../googleplay.csv", index_col='package')
df = df_simple.join(df_googleplay, on="app_id")
df_sonar = pandas.read_csv("../results_sonar.csv", index_col='package')
df_sonar.fillna(0, inplace=True)
df_sonar = df_sonar.add_prefix('sonar_')
df = df.join(df_sonar, on="app_id")

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

df['unit_tests'] = df[unit_test_frameworks].apply(any, axis=1)
projects_with_unit_tests = df['unit_tests'].sum()
df['ui_tests'] = df[ui_automation_frameworks].apply(any, axis=1)
projects_with_ui_tests = df['ui_tests'].sum()
df["ci/cd"] = df[ci_services].apply(any, axis=1)

hall_of_fame = df[df[['ci/cd', 'unit_tests', 'ui_tests']].all(axis=1)].sort_values('stars', ascending=False)
hall_of_fame.to_csv('hall_of_fame.csv')

