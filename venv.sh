#!/usr/bin/env bash
prefix=env
rm -r $prefix
virtualenv --python=/usr/bin/python3 $prefix
source $prefix/bin/activate
pip install --ignore-installed --no-cache-dir --upgrade -r requirements.txt
