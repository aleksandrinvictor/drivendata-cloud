MODEL=$1

mkdir benchmark_src && mkdir benchmark_src/cloud && mkdir benchmark_src/assets
cp cloud/*.py benchmark_src/cloud/ && cp -r cloud/models benchmark_src/cloud/
cp -r assets/$MODEL benchmark_src/assets
cp -r pretrained_models/ benchmark_src/pretrained_models
cp main.py benchmark_src/

cd benchmark_src; zip -r ../sub_$MODEL.zip *
