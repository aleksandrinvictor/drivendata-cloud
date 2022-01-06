set -e

for fold in 0 1 2 3 4
do
    python cloud/train.py --fold=$fold
done
