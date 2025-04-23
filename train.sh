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