#!/bin/bash

set -e

nbconvert=nbconvert

for F in ../ipynb/*.ipynb; do
    echo "Converting to html:" \"$F\"
    ${nbconvert} --format html "$F"
done

mkdir -p nbpages/files/images 

set +e
mv ../ipynb/*.html nbpages/
mv ../ipynb/*.png nbpages/
set -e
cp ../ipynb/images/* nbpages/files/images/

