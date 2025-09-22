cuda_0:
	CUDA_VISIBLE_DEVICES=0 dvc repro

cuda_1:
	CUDA_VISIBLE_DEVICES=1 dvc repro

cuda_2:
	CUDA_VISIBLE_DEVICES=2 dvc repro

cuda_3:
	CUDA_VISIBLE_DEVICES=3 dvc repro
