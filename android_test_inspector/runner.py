"""Script that puts everything together and runs experiments."""

from google_play_reader.models import AppDatabase
import os
import csv

from android_test_inspector.crawler import download_fdroid, parse_fdroid, analyze_project, INSPECTORS

FDROID_DATA_FILENAME = "./fdroid.xml"
PROJECTS_DATA_CSV = "./fdroid_repos.csv"
RESULTS_CSV = "./results.csv"
GOOGLEPLAY_CSV = "./googleplay.csv"
CACHE_FDROID = True

if __name__ == "__main__":
    if not os.path.isfile(FDROID_DATA_FILENAME) or not CACHE_FDROID:
        download_fdroid(FDROID_DATA_FILENAME)

    if not os.path.isfile(PROJECTS_DATA_CSV) or os.path.getctime(PROJECTS_DATA_CSV) < os.path.getctime(FDROID_DATA_FILENAME):
        parse_fdroid(
            file_in=FDROID_DATA_FILENAME,
            file_out=PROJECTS_DATA_CSV,
        )

    if not os.path.isfile(RESULTS_CSV) or os.path.getctime(RESULTS_CSV) < os.path.getctime(PROJECTS_DATA_CSV):
        with open(PROJECTS_DATA_CSV, 'r') as projects_csvfile:
            csv_reader = csv.DictReader(projects_csvfile)
            fieldnames = list(csv_reader.fieldnames) + list(INSPECTORS.keys())
            with open(RESULTS_CSV, 'w') as results_csvfile:
                csv_writer = csv.DictWriter(results_csvfile, fieldnames=fieldnames)
                csv_writer.writeheader()
                for row in csv_reader:
                    print(row)
                    project_results = analyze_project(row['github_link'], "./tmp/{user}_{project_name}".format(**row))
                    row.update(project_results)
                    csv_writer.writerow(row)

    #collect information from Google Play
    if not os.path.isfile(GOOGLEPLAY_CSV) or os.path.getctime(GOOGLEPLAY_CSV) < os.path.getctime(RESULTS_CSV):
        with open(RESULTS_CSV, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            packages = [row['app_id'] for row in csv_reader]
        google_play_center = AppDatabase(GOOGLEPLAY_CSV)
        google_play_center.bulk_process(packages)
