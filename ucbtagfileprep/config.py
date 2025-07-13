import os

from kkconfig import local

local.load_settings(os.environ.get('KKCONFIG_PATH','.config.yaml'),globals())