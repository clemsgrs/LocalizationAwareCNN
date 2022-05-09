mkdir ~/.kaggle
mv "$1" ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d clemsy/bracs-train-small
unzip -qq bracs-train-small.zip -d "data/"
rm bracs-train-small.zip

# kaggle datasets download -d clemsy/bracs-train-small-supplementary
# unzip -qq bracs-train-small-supplementary.zip -d "data/"
# rm bracs-train-small-supplementary.zip

# kaggle datasets download -d clemsy/bracs-val-small
# unzip -qq bracs-val-small.zip -d "data/"
# rm bracs-val-small.zip

# kaggle datasets download -d clemsy/bracs-test-small
# unzip -qq bracs-test-small.zip -d "data/"
# rm bracs-test-small.zip