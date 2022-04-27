pip install kaggle

PATH_TO_KAGGLE_JSON=$1

mkdir ~/.kaggle
mv PATH_TO_KAGGLE_JSON ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d clemsy/bracs-train-small
unzip -qq -j bracs-train-small.zip -d "data/"