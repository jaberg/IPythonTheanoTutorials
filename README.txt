README OF GH-PAGES BRANCH OF IPYTHON THEANO TUTORIALS
=====================================================

The pages published for the ipython notebooks works like this:

* index.html is auto-built using the github pages wizard thing

* All other html files are built from ipynb files in the 'master' branch


## Rebuilding index.html

Rebuild the index.html file by editing the index.html directly.
DO NOT regenerate it using the github wizard, it will erase everything else!!


## Rebuilding all the notebook html files

1. Install nbconvert (https://github.com/ipython/nbconvert)

2. Set up this project branch in a subfolder (recommended: "html") of the
IPythonTheanoTutorials master branch.

3. run ./rebuild_pages.sh 

4. commit the changed html pages to [this] gh-pages branch to make them appear
on the website.


