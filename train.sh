CUDA_VISIBLE_DEVICES='0' python3 -u train_SFace_torch.py --workers_id 0 --batch_size 256 --lr 0.1 --stages 50,70,80 --data_root ./datasets/train/faces_webface_112x112 --eval_path ./datasets/eval --target lfw --outdir ./results/mobilefacenet-sface-casia --param_a 0.87 --param_b 1.2 2>&1|tee ./logs/mobilefacenet-sfacce-casia.log

CUDA_VISIBLE_DEVICES='0,1' python3 -u train_SFace_torch_freeze_backbone.py \
    --workers_id 0,1 \
    --batch_size 256 \
    --lr 0.1 \
    --stages 50,70,80 \
    --data_root /kaggle/input/casia-webface/faces_webface_112x112 \
    --eval_path /kaggle/input/casia-webface/faces_webface_112x112 \
    --resume_backbone weights/face_recognition_sface_2021dec.pth
    --target lfw \
    --outdir ./results/mobilefacenet-sface-casia \
    --param_a 0.87 \
    --param_b 1.2 2 > &1 | tee ./logs/mobilefacenet-sfacce-casia.log