env=$CONDA_DEFAULT_ENV
if [ -z "$env" ]
then 
    source activate pytorch1.2
fi

gpu=$1
if [ -z "$gpu" ]
then
    gpu=0
fi

CUDA_VISIBLE_DEVICES=$gpu python /data/jinhuama/gy/UDA/main.py