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
#set -e

for F in nbpages/*.html ; do
    NEWF=$(echo "$F" | sed -e "s/(/_/g" -e "s/)/_/g")
    echo "Sanitizing URL: $F -> $NEWF"
    git mv "$F" "$NEWF"
done

cp ../ipynb/images/* nbpages/files/images/

