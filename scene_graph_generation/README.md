
This is the scene graph generation part of our codes for our ICPR 2022 paper. 

You can get more information from our ICPR 2022 paper and our supplementaty material pdf file.
You can get big files in our google drive https://drive.google.com/drive/folders/1tYpC70ZMT10vTbrjur1JcD1ftsu2XyE5?usp=sharing
We developed this repository based on an open-source SGG Benchmark Project: https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


## Our hardware configuration and CUDA Version

	CPU: Intel(R) Xeon(R) Gold 5222 CPU @ 3.80GHz
	GPU: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti] 
	CUDA Toolkit Version: Release 11.1, V11.1.105


## Reproduce our results
You can reproduce our results by the following commands:

Predicate classification training:

	CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10000 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 44000 SOLVER.VAL_PERIOD 10000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.PRETRAINED_DETECTOR_CKPT /home/<USER>/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/<USER>/checkpoints/motif-precls-exmp-na SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.EMBED_DIM 200 GLOVE_DIR /home/<USER>/glove/na

Predicate classification test:

	CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10010 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/<USER>/glove/na MODEL.PRETRAINED_DETECTOR_CKPT /home/<USER>/checkpoints/motif-precls-exmp-na OUTPUT_DIR /home/<USER>/checkpoints/motif-precls-exmp-na MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.EMBED_DIM 200

Scene graph classification training: 

	CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --master_port 10000 --nproc_per_node=1 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 50000 SOLVER.VAL_PERIOD 10000 SOLVER.CHECKPOINT_PERIOD 2000 GLOVE_DIR /home/<USER>/glove/na MODEL.PRETRAINED_DETECTOR_CKPT /home/<USER>/checkpoints/pretrained_faster_rcnn/model_final.pth SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.EMBED_DIM 200  OUTPUT_DIR /home/<USER>/checkpoints/vctree-sgcls-exmp-na SOLVER.BASE_LR 0.0001

Scene graph classification test: 

	CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10010 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/<USER>/glove/na MODEL.PRETRAINED_DETECTOR_CKPT /home/<USER>/checkpoints/vctree-sgcls-exmp-na OUTPUT_DIR /home/<USER>/checkpoints/vctree-sgcls-exmp-na MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.EMBED_DIM 200


Scene graph detection training: 

	CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10000 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 38000 SOLVER.VAL_PERIOD 10000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.PRETRAINED_DETECTOR_CKPT /home/<USER>/checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR /home/<USER>/checkpoints/motif-sggen-exmp-na SOLVER.PRE_VAL False MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.EMBED_DIM 200 GLOVE_DIR /home/<USER>/glove/na SOLVER.BASE_LR 0.001

Scene graph detection test: 

	CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/<USER>/glove/na MODEL.PRETRAINED_DETECTOR_CKPT /home/<USER>/checkpoints/motif-sggen-exmp-na OUTPUT_DIR /home/<USER>/checkpoints/motif-sggen-exmp-na MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.EMBED_DIM 200

## Download our trained model.
If you do not want to train the model by yourself, you can directly download our trained model *-*-exmp-na.zip in our google drive, unzip them and move it to /home/<USER>/checkpoints/ .

	motif-precls-exmp-na: Our Motif-RGN model for the predicate classification task.
	vctree-sgcls-exmp-na: Our VCT-RGN model for the scene graph classification task.
	motif-sggen-exmp-na: Our Motif-RGN model for the scene graph detection task.
	
Note that you have to edit the file /home/<USER>/checkpoints/<MODEL>/last_checkpoint to change the username.
