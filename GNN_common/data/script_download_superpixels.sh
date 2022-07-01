

# Command to download dataset:
#   bash script_download_superpixels.sh

DIR=/home/jxu8/Code/federated_learning_jx/federated_learning/GNN_common/data/superpixels
cd $DIR


FILE=MNIST.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/wcfmo4yvnylceaz/MNIST.pkl?dl=1 -o MNIST.pkl -J -L -k
fi


FILE=CIFAR10.pkl
if test -f "$FILE"; then
	echo -e "$FILE already downloaded."
else
	echo -e "\ndownloading $FILE..."
	curl https://www.dropbox.com/s/agocm8pxg5u8yb5/CIFAR10.pkl?dl=1 -o CIFAR10.pkl -J -L -k
fi






