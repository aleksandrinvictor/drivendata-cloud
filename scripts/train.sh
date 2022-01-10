set -e
CFG_PATH=$1

for fold in 0 1 2 3 4
do
    python cloud/train.py --fold=$fold --cfg_path=$CFG_PATH
done
