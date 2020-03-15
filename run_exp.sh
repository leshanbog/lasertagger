TRAIN_DATASET_VOCAB_OPT='/data/alolbuhtijarov/dataset/ria.shuffled.train.json'
FORMAT_OPT='ria'

TRAIN_DATASET_PREPROC='/data/alolbuhtijarov/dataset/ria.shuffled.train.json'
FORMAT_PREPROC='ria'

BERT_DIR='/data/alolbuhtijarov/model/rubert_cased_L-12_H-768_A-12_v2'

OUTPUT_DIR='/data/alolbuhtijarov/first_experiment'

TEST_DATASET='/data/alolbuhtijarov/dataset/ria.shuffled.test_2k.json'


set -x


mkdir -p model;

python3 phrase_vocabulary_optimization.py --input_file="$TRAIN_DATASET_VOCAB_OPT" --input_format="$FORMAT_OPT" --vocabulary_size=4092 --output_file=model/label_map.txt;


# python3 helpers/glue_phrases_script.py model/label_map.txt model/label_map.txt;


python3 preprocess_main.py --input_file="$TRAIN_DATASET_PREPROC" --input_format="$FORMAT_PREPROC" --output_tfrecord=model/train.tf_record --label_map_file=model/label_map.txt --vocab_file="$BERT_DIR"/vocab.txt --output_arbitrary_targets_for_infeasible_examples=false;


python3 run_lasertagger.py --training_file=model/train.tf_record --label_map_file=model/label_map.txt --model_config_file=configs/lasertagger_config.json --output_dir="$OUTPUT_DIR" --do_train=true --num_train_epochs=5 --train_batch_size=4 --num_train_examples=$(cat model/train.tf_record.num_examples.txt) --init_checkpoint="$BERT_DIR"/bert_model.ckpt --save_checkpoints_steps=30000;


python3 run_lasertagger.py --label_map_file=model/label_map.txt --model_config_file=configs/lasertagger_config.json --output_dir="$OUTPUT_DIR" --do_export=true --export_path="$OUTPUT_DIR"/export;

python3 predict_main.py --input_file="$TEST_DATASET" --input_format=ria --output_file=model/ria_test_pred.tsv --label_map_file=model/label_map.txt --vocab_file="$BERT_DIR"/vocab.txt --saved_model="$OUTPUT_DIR"/export/$(ls "$OUTPUT_DIR"/export);

python3 helpers/calc_metrics.py model/ria_test_pred.tsv;

set +x

