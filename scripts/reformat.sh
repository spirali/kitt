#!/bin/bash

DIRECTORIES="kitt tests"

cd `dirname $0`/..
isort --profile black ${DIRECTORIES}
black ${DIRECTORIES}
flake8 ${DIRECTORIES} --ignore=E203,W503
