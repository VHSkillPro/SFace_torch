CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch.py \
    --workers_id 0 \
    --batch_size 256 \
    --lr 0.1 \
    --stages 50,70,80 \
    --data_root ./datasets/train \
    --eval_path ./datasets/eval \
    --target lfw \
    --outdir ./results/mobilefacenet-sface-casia \
    --param_a 0.87 \
    --param_b 1.2 2>&1|tee ./logs/mobilefacenet-sface-casia.log

CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch_freeze_backbone.py \
    --workers_id 0 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.1 \
    --stages 20,30,40 \
    --data_root datasets/train/downscale_casia-webface_2_converted \
    --eval_path datasets/eval \
    --resume_backbone weights/face_recognition_sface_2021dec.pth \
    --target lfw,cplfw,cfp_fp,cfp_ff,calfw,agedb_30 \
    --outdir ./results/mobilefacenet_frezze-sface-casia_downscale \
    --param_a 0.87 \
    --param_b 1.2 2>&1 | tee ./logs/mobilefacenet_frezze-sface-casia_downscale.log

CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch_freeze_head.py \
    --workers_id 0 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.1 \
    --stages 20,30,40 \
    --data_root datasets/train/downscale-casia_webface-2-converted \
    --eval_path datasets/eval \
    --resume_head weights/Head_SFaceLoss.pth \
    --resume_backbone weights/ \
    --target lfw,cplfw,cfp_fp,cfp_ff,calfw,agedb_30 \
    --outdir ./results/mobilefacenet-sface_frezze-casia_downscale \
    --param_a 0.87 \
    --param_b 1.2 2>&1 | tee ./logs/mobilefacenet-sface_freeze-casia_downscale.log

CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch_KD_CS.py \
    --workers_id 0 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.1 \
    --stages 20,30,40 \
    --data_root datasets/train/downscale-casia_webface-2-converted \
    --eval_path datasets/eval \
    --target lfw,cplfw,cfp_fp,cfp_ff,calfw,agedb_30 \
    --outdir ./results/mobilefacenet-sface_KD_CS-casia_downscale \
    --param_a 0.87 \
    --teacher_backbone weights/face_recognition_sface_2021dec.onnx \
    --resume_head weights/Head_SFaceLoss.pth \
    --resume_backbone weights/ \
    --param_b 1.2 2>&1 | tee ./logs/mobilefacenet-sface_KD_CS-casia_downscale.log

CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch_KD_MSE.py \
    --workers_id 0 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.1 \
    --stages 20,30,40 \
    --data_root datasets/train/downscale-casia_webface-2-converted \
    --eval_path datasets/eval \
    --target lfw,cplfw,cfp_fp \
    --outdir ./results/mobilefacenet-sface_KD_MSE-casia_downscale \
    --param_a 0.87 \
    --teacher_backbone weights/face_recognition_sface_2021dec_extend.onnx \
    --param_b 1.2 2>&1 | tee ./logs/mobilefacenet-sface_KD_MSE-casia_downscale.log

CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch_KD_MSE+CS.py \
    --workers_id 0 \
    --batch_size 256 \
    --epochs 50 \
    --lr 0.1 \
    --stages 20,30,40 \
    --data_root datasets/train/downscale-casia_webface-2-converted \
    --eval_path datasets/eval \
    --target lfw,cplfw,cfp_fp \
    --outdir ./results/mobilefacenet-sface_KD_MSE+CS-casia_downscale \
    --param_a 0.87 \
    --teacher_backbone weights/face_recognition_sface_2021dec_extend.onnx \
    --param_b 1.2 2>&1 | tee ./logs/mobilefacenet-sface_KD_MSE+CS-casia_downscale.log