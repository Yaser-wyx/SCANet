cd ../

ckpt="/home/yaser/Projects/iros_code/code/Hyperplane_Assembling/ECCV_exp/lightning_logs/cfg_version2_raw_SCANet_for_150test_2_@0-all_D@train_valid@continue/checkpoints/epoch=99-val_loss_sum=5.30-val_trans_acc=66.49.ckpt"
data_root=/home/yaser/Projects/iros_code/SCANet_data/LEGO-ECA/train
test_json=/home/yaser/Projects/iros_code/SCANet_data/LEGO-ECA/train/train_for_test_150.json

python eval_with_correction.py --checkpoints_dir ./checkpoints/ --config_path ./configs/SCANet.yaml \
      --checkpoint_path $ckpt --dataroot $data_root  --set_for_test_path $test_json  \
      --name mepnet --model hourglass_shape_cond --dataset_mode legokps_shape_cond --batch_size 10 --epoch latest --num_threads 15 --occ_out_channels 8 \
      --occ_fmap_size 256 --load_bbox --load_conn --top_center --load_bricks --serial_batches --dataset_alias synthetic_train_autoreg --camera_jitter 0 --max_objs 50 \
      --max_brick_types 5 --n_vis 200 --crop_brick_occs --num_bricks_single_forward 5 --kp_sup --num_stacks 2 --cbrick_brick_kp --img_type laplacian --load_mask --max_dataset_size 300 \
      --load_mask --search_ref_mask --allow_invisible --oracle_percentage 0.0 --symmetry_aware_rotation_label --no_coordconv --autoregressive_inference --output_pred_json --predict_masks --n_set 1 --visualize
