# code

## preprocessing
download code to Lian_project_code/
 
## train MUNIT
cd MUNIT
python train_pelvic.py --gpu {GPU_ID} --data_dir {DATA_DIR} --log_dir {LOG_DIR} --checkpoint_dir {CHECKPOINT_DIR} --logfile {LOGFILE_NAME}

## test MUNIT
python test_pelvic.py --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR} --output_dir {OUTPUT_DIR}


## train UNIT
cd MUNIT
python train_pelvic.py --trainer UNIT --gpu {GPU_ID} --data_dir {DATA_DIR} --log_dir {LOG_DIR} --checkpoint_dir {CHECKPOINT_DIR} --logfile {LOGFILE_NAME}

## test UNIT
python test_pelvic.py --trainer UNIT --gpu {GPU_ID} --data_dir {DATA_DIR} --checkpoint_dir {CHECKPOINT_DIR} --output_dir {OUTPUT_DIR}
