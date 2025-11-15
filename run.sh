python p1_generate_dataset.py

python p2_get_activation.py --llm_name Qwen2.5-3B-Instruct --mode train --gpu 0
python p2_get_activation.py --llm_name Qwen2.5-3B-Instruct --mode test --gpu 0

# python p3_clustering.py --llm_name Qwen2.5-3B-Instruct --n_clusters 5
# python p3_clustering.py --llm_name Qwen2.5-3B-Instruct --n_clusters 10
# python p3_clustering.py --llm_name Qwen2.5-3B-Instruct --n_clusters 15
python p3_clustering.py --llm_name Qwen2.5-3B-Instruct --n_clusters 20

# python p4_iter.py --llm_name Qwen2.5-3B-Instruct --n_clusters 5
# python p4_iter.py --llm_name Qwen2.5-3B-Instruct --n_clusters 10
# python p4_iter.py --llm_name Qwen2.5-3B-Instruct --n_clusters 15
python p4_iter.py --llm_name Qwen2.5-3B-Instruct --n_clusters 20

# python p5_result_objective.py --llm_name Qwen2.5-3B-Instruct --n_clusters 5
# python p5_result_objective.py --llm_name Qwen2.5-3B-Instruct --n_clusters 10
# python p5_result_objective.py --llm_name Qwen2.5-3B-Instruct --n_clusters 15
python p5_result_objective.py --llm_name Qwen2.5-3B-Instruct --n_clusters 20

# python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 5 --classifier svc --gpu 0
# python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 10 --classifier svc --gpu 0
# python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 15 --classifier svc --gpu 0
python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 20 --classifier svc --gpu 0
# python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 5 --classifier lr --gpu 0
# python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 10 --classifier lr --gpu 0
# python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 15 --classifier lr --gpu 0
python p6_result_prediction.py --llm_name Qwen2.5-3B-Instruct --n_clusters 20 --classifier lr --gpu 0
