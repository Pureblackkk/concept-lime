python ./bias-extractor/main.py \
--save_keyword true \
--save_keyword_dir ~/concept-lime/keyword/waterbird_gpt_cluster \
--skip_bt2 true \
--b2t_dir ~/b2t-master \
--b2t_dataset waterbird \
--b2t_model best_model_Waterbirds_erm.pth \
--b2t_keyword_res_dir ./diff/ \
--concept_cluster