pip install kaggle

mkdir ~/.kaggle
mv "$1" ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d clemsy/bracs-train-small
unzip -qq bracs-train-small.zip -d "data/"
rm bracs-train-small.zip