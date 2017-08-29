from git import Repo
import click
from inspector import inspectors

def _gitclone(repo_link, clone_dir):
    try:
        return Repo.clone_from(repo_link, clone_dir)
    except Exception as e:
        if "is not an empty directory" in str(e):
            click.secho('Failed cloning {}: directory is not empty.'.format(repo_link), fg='yellow', err=True)
        else:
            click.secho('Failed cloning {}.'.format(repo_link), fg='red', err=True)
        return None

def analyze_project(repo_link, clone_dir, use_local_files=True, keep_files=True):
    repo = _gitclone(repo_link, clone_dir)
    if repo or use_local_files: 
        try:
            return {
                inspector_name: inspector.check(clone_dir)
                for (inspector_name, inspector) in inspectors.items()
            } 
        finally:           
            if repo and not keep_files:
                 #delete files
                 pass
    else:
        return None
