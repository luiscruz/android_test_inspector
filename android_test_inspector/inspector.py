"""Matchers to find test suite usage in Android projects."""

import os
import fnmatch
import re
import abc

# pylint: disable=too-few-public-methods
# pylint: disable=invalid-name

_FILE_IGNORE = [
    '*.jar',
    '*.aar',
    '*.zip',
    'index',
    '*.idx',
    '*.pack',
    '*.png',
    '*.jpg',
    '*.bmp',
    '*.odg',
    '*.xcf',
    '*.apk',
    '*.mp3',
    '*.mka',
    '*.ogg',
    '*.opus',
    '*.wav',
    '*.pdf',
    '*.ttf',
    '*.svgz',
    '*.woff',
    '*.dat',
    '*.o',
    '*.db',
    '*.odp',
    '*.psd',
    '*.bin',
    '.DS_Store'.lower(),
    'ChangeLog'.lower(),
    '*.jks',
    '*.webp',
]

class Inspector(object):
    """Abstract class to store knowledge to assess whether a project uses a test suite."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def check(self, root_dir):
        """Abstract method to check whether given project uses this test suite."""
        return

class InspectorRegex(Inspector):
    """Inspector using regex expression for files and content."""
    def __init__(self, files_pattern, framework_pattern):
        self.files_pattern = files_pattern
        self.framework_pattern = framework_pattern

    def check(self, root_dir):
        """Check whether given project uses this test suite."""
        for dirpath, _, files in os.walk(root_dir):
            for file_matched in fnmatch.filter(files, self.files_pattern):
                if any([fnmatch.fnmatch(file_matched.lower(), pattern) for pattern in _FILE_IGNORE]):
                    continue
                try:
                    with open(os.path.join(dirpath, file_matched), 'rt') as file_opened:
                        try:
                            content_as_string = file_opened.read()
                            match = re.search(self.framework_pattern, content_as_string)
                            if match:
                                return True
                        except UnicodeDecodeError:
                            print("Error reading file {}".format(file_matched))
                            continue
                except FileNotFoundError:
                    print("File not found (skipping): {}".format(file_matched))
                    continue
                except:
                    print("Unknown error in file {}".format(file_matched))
                    continue
        return False

class InspectorComposer(Inspector):
    """Inspector to combine multiple inspectors."""
    def __init__(self, *inspectors):
        self.inspectors = inspectors

    def check(self, root_dir):
        """Check whether given project uses this test suite."""
        return any(inspector.check(root_dir) for inspector in self.inspectors)

#UI Automation frameworks
inspector_androidviewclient = InspectorComposer(
    InspectorRegex("requirements.txt", "androidviewclient"),
    InspectorRegex("*.py", "com.dtmilano.android.viewclient"),
)
inspector_appium = InspectorComposer(
    InspectorRegex("requirements.txt", "Appium-Python-Client"),
    InspectorRegex("*.py", "appium"),
    InspectorRegex("appium.txt", ""),
    InspectorRegex("Gemfile", "appium"),
    InspectorRegex("pom.xml", "io.appium"),
    InspectorRegex("*gradle*", "io.appium"),
)
inspector_calabash = InspectorComposer(
    InspectorRegex("Gemfile", "calabash"),
)
inspector_espresso = InspectorRegex(
    "*gradle*",
    "espressoVersion|com.android.support.test.espresso"
)
inspector_monkeyrunner = InspectorRegex("*.py", "MonkeyRunner|MonkeyDevice")
inspector_pythonuiautomator = InspectorRegex("*.py", "uiautomator")
inspector_robotium = InspectorRegex("*gradle*", "com.jayway.android.robotium")
inspector_uiautomator = InspectorRegex(
    "*gradle*",
    "uiautomatorVersion|com.android.support.test.uiautomator"
)
inspector_projectquantum = InspectorRegex("pom.xml", "com.quantum")
inspector_qmetry = InspectorRegex("*.xml", "com.qmetry")
# Cloud testing services
inspector_saucelabs = InspectorComposer(
    InspectorRegex("*.py", "ondemand.saucelabs.com"),
    InspectorRegex("*.java", "ondemand.saucelabs.com"),
    InspectorRegex("*.kt", "ondemand.saucelabs.com"),
    InspectorRegex("*.js", "ondemand.saucelabs.com"),
    InspectorRegex("*", "SAUCE_USERNAME"),
)
inspector_firebase = InspectorRegex("*gradle*", "com.google.firebase")
inspector_perfecto = InspectorComposer(
    InspectorRegex("*.java", ".perfectomobile.com"),
    InspectorRegex("*.kt", ".perfectomobile.com"),
)
inspector_bitbar = InspectorComposer(
    InspectorRegex("*.py", "TESTDROID|testdroid_utils|testdroid.com"),
    InspectorRegex("*.rb", "TESTDROID|testdroid"),
    InspectorRegex("pom.xml", "com.testdroid"),
)
# Unit test frameworks
inspector_junit = InspectorRegex("*gradle*", "junit:junit")
inspector_androidjunitrunner = InspectorRegex("*gradle*", "android.support.test.runner.AndroidJUnitRunner")
inspector_roboelectric = InspectorRegex("*gradle*", "org.robolectric:robolectric")
inspector_robospock = InspectorRegex("*gradle*", "org.robospock:robospock")
inspector_travis = InspectorRegex("travis.yml", "")
inspector_circleci = InspectorRegex("circle.yml", "")
inspector_codeship = InspectorRegex("codeship*.yml", "")
inspector_codefresh = InspectorRegex("codefresh.yml", "")

INSPECTORS = {
    "androidviewclient": inspector_androidviewclient,
    "appium": inspector_appium,
    "calabash": inspector_calabash,
    "espresso": inspector_espresso,
    "monkeyrunner": inspector_monkeyrunner,
    "pythonuiautomator": inspector_pythonuiautomator,
    "robotium": inspector_robotium,
    "uiautomator": inspector_uiautomator,
    "projectquantum": inspector_projectquantum,
    "qmetry": inspector_qmetry,
    # Cloud testing services
    "saucelabs": inspector_saucelabs,
    "firebase": inspector_firebase,
    "perfecto": inspector_perfecto,
    "bitbar": inspector_bitbar,
    # CI (https://www.code-maze.com/top-8-continuous-integration-tools/)
    "travis": inspector_travis,
    'circleci': inspector_circleci,
    'codeship': inspector_codeship,
    # teamcity: does not store conf files in the repo
    # jenkins: does not store conf files in the repo
    # go cd: does not store conf files in the repo
    # bamboo: does not store conf files in the repo
    # gitlab: requires gitlab project
    'codefresh': inspector_codefresh
    #unit test frameworks
    "junit": inspector_junit,
    "androidjunitrunner": inspector_androidjunitrunner,
    "roboelectric": inspector_roboelectric,
    "robospock": inspector_robospock,
}
