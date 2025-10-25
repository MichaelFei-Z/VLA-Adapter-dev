data_name=galbot21b_spider_board
current_time=$(date +"%Y-%m-%d_%H-%M-%S")
output_dir=/home/chenghan/workspace/fei/model_ckpt/vla_adapter
data_root_dir=/home/data_sda/galbot21b_rlds

CUDA_VISIBLE_DEVICES=2,3,5,6 torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
--vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
--config_file_path pretrained_models/configs \
--data_root_dir $data_root_dir \
--dataset_name $data_name \
--run_root_dir $output_dir/ckpt \
--use_film False \
--num_images_in_input 2 \
--use_proprio True \
--use_lora True \
--use_fz False \
--use_minivlm True \
--image_aug True \
--num_steps_before_decay 200000 \
--max_steps 200005 \
--save_freq 10000 \
--save_latest_checkpoint_only False \
--merge_lora_during_training True \
--batch_size 4 \
--grad_accumulation_steps 2 \
--learning_rate 2e-4 \
--lora_rank 64 \
--use_pro_version True \
--wandb_entity "1577903056-east-china-university-of-science-and-technology" \
--wandb_project "vla-adapter-dev" \
--run_id_note $data_name--$current_time \
2>&1 | tee $output_dir/logs/$data_name--$current_time.log