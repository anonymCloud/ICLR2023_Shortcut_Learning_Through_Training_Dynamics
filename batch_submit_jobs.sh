# copmute PD of test images wrt a model
# use rtx6000 or v100-32, we need 32 GB or more for these expts
#!/bin/bash
partition=GPU-shared
num_nodes=1
gpu=gpu:v100-32:1
wall_time=48:00:00  # it takes 40 seconds to compute PD for a single image. (multiply by number of imgs in your test_csv)
                    # last job had 2k images and took 1.5 hrs to complete
mail_type=ALL


epochs=(0 205 410 615 820 1025 1230 1435 1640 1845 2050 2255 2460 2665 2870 3075 3280 3485 3690 3895 4100 4305 4510 4715 4920 5125)

for ep in ${epochs[@]}
do
    # user hyperparams
    code_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/scripts/compute_pd.py
    ckpt_path=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Output/NIH/imgsize_128/nih_shortcut_fine/e1-it$ep.pt
    pkl_path=
    test_csv=/xxx/home/xxx/xxxp/xxx/data/nih/train_splits/train_only_pneum_balanced2.csv
    num_imgs=2000   # number of images to randomly sample from test_csv for computing PD
    knn_pos_thresh=0.62
    knn_neg_thresh=0.38
    save_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/data/nih/pd_analysis/imgsize_128/nih_shortcut_iterations
    save_name=train_2k_ep1-it$ep
    job_name=$ep-nih-shortcut

    slurm_output=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
    slurm_err=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
    RUN_CMD="python $code_path $ckpt_path $pkl_path $test_csv $save_path $save_name --knn_pos_thresh $knn_pos_thresh --knn_neg_thresh $knn_neg_thresh --num_imgs $num_imgs"
    sbatch -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user xxx --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD
done 

# user hyperparams
code_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/scripts/temp_comp_pd.py
ckpt_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/medical_expts/nih/128/best_128-auc0.8326.pt
pkl_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/medical_expts/nih/128/nih_subset_embs.pkl
test_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/medical_expts/nih/128/intervene/nih8.csv
dataset=NIH
imgsize=128
num_imgs=7000   # number of images to randomly sample from test_csv for computing PD
knn_pos_thresh=0.62 #0.62
knn_neg_thresh=0.38 #0.38
save_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/medical_expts/nih/128/intervene/
save_name=nih8_pd
job_name=nih8_pd
df_path_col=path
cls_name=Pneumothorax

slurm_output=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
slurm_err=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
RUN_CMD="python $code_path $ckpt_path $pkl_path $test_csv $save_path $save_name --df_path_col $df_path_col --cls_name $cls_name --knn_pos_thresh $knn_pos_thresh --knn_neg_thresh $knn_neg_thresh --num_imgs $num_imgs --dataset $dataset --img_size $imgsize"
sbatch -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user xxx --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD




# Classifier training
#!/bin/bash
partition=BatComputer
num_nodes=1
gpu=gpu:rtx5000:1
wall_time=30:00:00 # for 200k samples of size 128x128 1 hour for 2 epochs
mail_type=ALL

# user hyperparams
code_path=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Train_Classifier_DenseNet.py
config_path=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Configs/Classifier/waterbirds/waterbirds.yaml
job_name=waterbirds_train

slurm_output=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
slurm_err=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
RUN_CMD="python $code_path --config $config_path"
sbatch -A bio170034p -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user xxx --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD





# Domino Dataset Training
#!/bin/bash
partition=BatComputer
num_nodes=1
gpu=gpu:rtx5000:1
wall_time=0:20:00 # for 200k samples of size 128x128 1 hour for 2 epochs
mail_type=ALL

code_path=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/train.py

datasets=(kmnistPatch)
seeds=(10 500 256 13 1597 546 21 262 438 496 555 657)
string1=_bot_kmnist_2class_ro_1p0
string2=_bot_kmnist
test_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_blank_bot_kmnist_2class_ro_1p0.csv
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        # user hyperparams
        train_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_$dataset$string1.csv    
        expt_name=resnet18_top_$dataset$string2$seed  
        job_name=resnet18_top_$dataset$string2$seed
        slurm_output=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
        slurm_err=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
        RUN_CMD="python $code_path --train_csv $train_csv --test_csv $test_csv --expt_name $expt_name --seed $seed"
        sbatch -A bio170034p -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user xxx --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD
    done 
done



datasets=(cifar10 svhn)
string1=_bot_fmnist_2class_ro_1p0
string2=_bot_fmnist
test_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_blank_bot_fmnist_2class_ro_1p0.csv
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        # user hyperparams
        train_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_$dataset$string1.csv    
        expt_name=resnet18_top_$dataset$string2$seed   
        job_name=resnet18_top_$dataset$string2$seed
        slurm_output=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
        slurm_err=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
        RUN_CMD="python $code_path --train_csv $train_csv --test_csv $test_csv --expt_name $expt_name --seed $seed"
        sbatch -A bio170034p -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user xxx --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD
    done 
done



datasets=(cifar10 svhn)
string1=_bot_kmnist_2class_ro_1p0
string2=_bot_kmnist
test_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_blank_bot_kmnist_2class_ro_1p0.csv
for seed in ${seeds[@]}
do
    for dataset in ${datasets[@]}
    do
        # user hyperparams
        train_csv=/xxx/home/xxx/xxxp/xxx/projects/shortcut_detection_and_mitigation/experiments/toy_expts/domino_expts/top_$dataset$string1.csv    
        expt_name=resnet18_top_$dataset$string2$seed   
        job_name=resnet18_top_$dataset$string2$seed
        slurm_output=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stdout
        slurm_err=/xxx/home/xxx/xxxp/xxx/projects/augmentation_by_explanation_eccv22/Jobs/slurm_output/$job_name.stderr
        RUN_CMD="python $code_path --train_csv $train_csv --test_csv $test_csv --expt_name $expt_name --seed $seed"
        sbatch -A bio170034p -p $partition -N $num_nodes --gres=$gpu -t $wall_time --mail-type $mail_type --mail-user xxx --output=$slurm_output --error=$slurm_err --job-name=$job_name ./submit_job.sh $RUN_CMD
    done 
done
