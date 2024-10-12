cd ../

export CUDA_VISIBLE_DEVICES="0"
your_dataset_path=/home/yaser/Projects/iros_code/SCANet_data # TODO replace by your dataset path
python  train_SCANet.py --dataset_root $your_dataset_path/LEGO-ECA/train --config_path ./configs/SCANet.yaml   \
                        --dataset_name train_valid --use-cache --dataset_cache_name "LEGO-ECA"
