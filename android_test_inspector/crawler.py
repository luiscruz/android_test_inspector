"""Logic to analyze a single project"""

from git import Repo
import click
import xml.etree.cElementTree as ET
import codecs
import requests
import os
import json
from pygithub3 import Github, exceptions as gh_exceptions
from urlparse import urlparse
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

def download_fdroid(filename):
    """Fetch whole Fdroid data."""
    url = "https://f-droid.org/repo/index.xml"
    response = requests.get(url, stream=True)
    with open(filename, "wb") as handle:
        with click.progressbar(
            response.iter_content(),
            label='Downloading F-Droid metadata'
        ) as bar:
            for data in bar:
                handle.write(data)

_github_api = None
def get_github_api():
    # authenticate github api
    global _github_api
    if _github_api is None:
        with open('./config.json') as config_file:
            config = json.load(config_file)
        _github_api = Github(
            user=config.get('github_username'),
            token=config.get('github_token')
        )
    return _github_api

def get_github_info(username, project):
    gh = get_github_api()
    try:
        repo = gh.repos.get(user=username, repo=project)
        contributors = gh.repos.list_contributors(
            user=username,
            repo=project
        ).all()
        n_contributors = len(contributors)
        n_commits = sum(map(lambda x: x.contributions, contributors))
        return {
            "forks": repo.forks_count,
            "stars": repo.stargazers_count,
            "created_at": repo.created_at.strftime("%Y-%m-%d"),
            "contributors": n_contributors,
            "commits": n_commits,
        }
    except gh_exceptions.NotFound:
        click.secho(
            'Failed to collect data about {}/{}.'.format(username,project),
            fg='yellow', err=True
        )
        return {}

def parse_fdroid(file_in, file_out, limit=None, no_cache=False):
    """Get repos of Android apps available at F-Droid."""
    lines = list()
    non_github_repos = 0 #keep track of non github repos
    for _, element in ET.iterparse(file_in):
        if element.tag == "application":
            for source_node in element.iter('source'):
                repo_link = source_node.text
                last_updated = element.find("lastupdated").text
                app_id = element.find("id").text
                category = element.find("category").text
                # get only projects with github repo
                if repo_link:
                    if "github" in repo_link:
                        parse = urlparse(repo_link)
                        path_items = parse.path.strip().split('/')
                        if len(path_items) == 3:
                            user = path_items[1]
                            project_name = path_items[2]
                            github_info = get_github_info(user, project_name)
                            if github_info:
                                lines.append((
                                    last_updated,
                                    repo_link,
                                    user,
                                    project_name,
                                    app_id,
                                    category,
                                    str(github_info.get('stars')),
                                    str(github_info.get('contributors')),
                                    str(github_info.get('commits')),
                                    str(github_info.get('forks')),
                                    github_info.get('created_at'),
                                ))
                                non_github_repos -= 1
                non_github_repos += 1
        if limit and len(lines) >= limit:
            break
    print "Found {github_repos} github repos from a total of {total_repos}".format(
        github_repos=len(lines),
        total_repos=len(lines)+non_github_repos
    )
    print "Saving %d repositories to \"%s\"." % (len(lines), file_out)
    lines.sort()
    with codecs.open(file_out, "w") as fo:
        fo.write("last_updated,github_link,user,project_name,app_id,category,stars,contributors,commits,forks,created_at\n")
        fo.writelines("\n".join([",".join(line) for line in lines[::-1]]))
        fo.write("\n")

