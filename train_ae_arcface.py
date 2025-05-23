import numpy as np

np.bool = np.bool_

import os
import time
import torch
import argparse
import onnx2torch
from torch import nn, optim
from util.utils import AverageMeter
from image_iter_rec import FaceDataset
from backbone.model_ae import AutoEncoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training AutoEncoder for ArcFace Model"
    )
    parser.add_argument(
        "--workers_id",
        type=str,
        default="0",
        help="GPU ID to use for training, e.g., '0,1' for multiple GPUs",
    )
    parser.add_argument("--lr", help="learning rate", default=0.1, type=float)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training the model",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the training data",
    )
    args = parser.parse_args()

    configuration = dict()
    if args.workers_id == "cpu" or not torch.cuda.is_available():
        configuration["GPU_ID"] = []
        print("check", args.workers_id, torch.cuda.is_available())
    else:
        configuration["GPU_ID"] = [int(i) for i in args.workers_id.split(",")]
    if len(configuration["GPU_ID"]) == 0:
        configuration["DEVICE"] = torch.device("cpu")
        configuration["MULTI_GPU"] = False
    else:
        configuration["DEVICE"] = torch.device("cuda:%d" % configuration["GPU_ID"][0])
        if len(configuration["GPU_ID"]) == 1:
            configuration["MULTI_GPU"] = False
        else:
            configuration["MULTI_GPU"] = True

    DEVICE = configuration["DEVICE"]
    MULTI_GPU = configuration["MULTI_GPU"]  # flag to use multiple GPUs
    GPU_ID = configuration["GPU_ID"]
    print("GPU_ID", GPU_ID)

    # ---------------------------- Begin - Load Data ----------------------------

    dataset = FaceDataset(
        path_imgrec=os.path.join(args.data_dir, "train.rec"), rand_mirror=True
    )
    trainloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    # ---------------------------- End - Load Data ----------------------------

    # ---------------------------- Begin - Load Model and Loss Function ----------------------------
    model = AutoEncoder()
    loss_function = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    arcface = onnx2torch.convert("weights/w600k_r50.onnx")

    if MULTI_GPU:
        model = nn.DataParallel(model, device_ids=GPU_ID)
        arcface = nn.DataParallel(arcface, device_ids=GPU_ID)
        arcface.to(DEVICE)
        model.to(DEVICE)
    else:
        arcface.to(DEVICE)
        model.to(DEVICE)

    print("========================== AutoEncoder ==========================")
    print("Model: ", model)
    print("========================= Loss Function =========================")
    print("Loss Function: ", loss_function)
    print("=========================== Optimizer ===========================")
    print("Optimizer: ", optimizer)
    print("============================ ArcFace ============================")
    print("ArcFace: ", arcface)
    print("=================================================================")
    # ---------------------------- End - Load Model and Loss Function ----------------------------

    # ---------------------------- Begin - Training Loop ----------------------------
    try:
        cosine_embedding_losses = AverageMeter()

        model.train()
        arcface.eval()
        for epoch in range(args.num_epochs):
            last_time = time.time()

            for inputs, labels in iter(trainloader):
                inputs = inputs.to(DEVICE)
                with torch.no_grad():
                    features = arcface.forward(inputs.float())

                model_outputs = model.forward(features)

                # Compute the loss
                loss = loss_function(
                    model_outputs,
                    features,
                    torch.ones(model_outputs.size(0)).to(DEVICE),
                )

                # Update the average loss
                cosine_embedding_losses.update(loss.mean().data.item(), inputs.size(0))

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print loss and speed
            print(
                "Epoch {} \t"
                "CosineEmbeddingLoss: {loss.avg:.4f}".format(
                    epoch + 1,
                    loss=cosine_embedding_losses,
                )
            )
            last_time = time.time()
            cosine_embedding_losses = AverageMeter()

    except Exception as e:
        raise e
    finally:
        # Save the model
        os.makedirs("results", exist_ok=True)
        if MULTI_GPU:
            torch.save(
                model.module.state_dict(),
                os.path.join("results", "model_ae_epoch_{}.pth".format(epoch + 1)),
            )
        else:
            torch.save(
                model.state_dict(),
                os.path.join("results", "model_ae_epoch_{}.pth".format(epoch + 1)),
            )

        dataset.close()
