mkdir ~/.kaggle
mv "$1" ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d clemsy/bracs-train-small-tensors
unzip -qq bracs-train-small-tensors.zip -d "data/"
rm bracs-train-small-tensors.zip

kaggle datasets download -d clemsy/bracs-train-small-supplementary-tensors
unzip -qq bracs-train-small-supplementary-tensors.zip -d "data/"
rm bracs-train-small-supplementary-tensors.zip

kaggle datasets download -d clemsy/bracs-val-small-tensors
unzip -qq bracs-val-small-tensors.zip -d "data/"
rm bracs-val-small.zip

kaggle datasets download -d clemsy/bracs-test-small-tensors
unzip -qq bracs-test-small-tensors.zip -d "data/"
rm bracs-test-small-tensors.zip