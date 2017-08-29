"""Logic to analyze a single project"""

from git import Repo
import click
from android_test_inspector.inspector import INSPECTORS

def _gitclone(repo_link, clone_dir):
    # pylint: disable=broad-except
    try:
        return Repo.clone_from(repo_link, clone_dir)
    except Exception as exception:
        if "is not an empty directory" in str(exception):
            click.secho(
                'Failed cloning {}: directory is not empty.'.format(repo_link),
                fg='yellow', err=True
            )
        else:
            click.secho('Failed cloning {}.'.format(repo_link), fg='red', err=True)
        return None

def analyze_project(repo_link, clone_dir, use_local_files=True, keep_files=True):
    """Analyze project given its git url."""
    repo = _gitclone(repo_link, clone_dir)
    if repo or use_local_files:
        try:
            return {
                inspector_name: inspector.check(clone_dir)
                for (inspector_name, inspector) in INSPECTORS.items()
            }
        finally:
            if repo and not keep_files:
                #delete files
                pass
    else:
        return None
