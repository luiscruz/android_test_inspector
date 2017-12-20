import json
import time

import click
from github import Github
from github.GithubException import GithubException


@click.command()
@click.argument('organization')
@click.argument('project')
def get_emails(organization, project):
    github = get_github_api()
    try:
        repo = github.get_user(organization).get_repo(project)
        contributors = list(repo.get_contributors())
        print(",".join("{}<{}>".format(c.name, c.email) for c in contributors if c.email))
    except GithubException:
        click.secho(
            'Failed to collect data about {}/{}.'.format(organization, project),
            fg='yellow', err=True
        )

_github_api = None
def get_github_api():
    # authenticate github api
    global _github_api
    if _github_api is None:
        with open('./config.json') as config_file:
            config = json.load(config_file)
        _github_api = Github(
            login_or_token=config.get('github_token')
        )
    return _github_api

def exit_gracefully(start_time):
    """Print time spent"""
    exit_time = time.time()
    duration = exit_time - start_time
    click.secho(
        "Execution took {:.4f} seconds.".format(duration),
        fg='blue'
    )

if __name__ == '__main__':
    start_time = time.time()
    try:
        get_emails()
    finally:
        exit_gracefully(start_time)
