"""Script that puts everything together and runs experiments."""
import os
from android_test_inspector.crawler import download_fdroid, parse_fdroid

FDROID_DATA_FILENAME = "./fdroid.xml"
PROJECTS_DATA_FILENAME = "./fdroid_repos.csv"
CACHE_FDROID = True

if __name__ == "__main__":
    if not os.path.isfile(FDROID_DATA_FILENAME) or not CACHE_FDROID:
        download_fdroid(FDROID_DATA_FILENAME)

    parse_fdroid(
        file_in=FDROID_DATA_FILENAME,
        file_out=PROJECTS_DATA_FILENAME,
    )
