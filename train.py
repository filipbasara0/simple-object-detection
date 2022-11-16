import argparse
import os
from datetime import datetime
from pprint import pformat

import torch
from torch.nn.utils import clip_grad_norm_
from torchinfo import summary

from tqdm.auto import tqdm

from utils import Logger
from datasets import dataset_factory
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from modeling import FCOS, FCOSPredictor
from utils import update_map, log_images

TRAINING_MAP_EPOCHS = 5
IMAGE_LOG_BATCHES = 10


def main(args):
    model = FCOS(num_classes=args.num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.pretrained_model_path:
        pretrained = torch.load(args.pretrained_model_path)
        model.load_state_dict(pretrained["model_state"])
        optimizer.load_state_dict(pretrained["optimizer_state"])

    fcos_predictor = FCOSPredictor(model, num_classes=args.num_classes)

    train_dataloader, eval_dataloader = dataset_factory(
        dataset_name=args.dataset_name,
        image_size=args.resolution,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size)

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        steps_per_epoch=len(train_dataloader),
        epochs=args.num_epochs)

    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    logs_path = f"./training_logs/{current_date}/"
    os.makedirs(logs_path, exist_ok=True)
    logger = Logger("simple-obj-det", file_path=f"{logs_path}/training_log.txt")
    model_summary = str(
        summary(model, (1, 3, args.resolution, args.resolution), verbose=1))
    logger.log_info(model_summary)

    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        losses_log = 0
        cls_losses_log = 0
        reg_losses_log = 0
        centerness_losses_log = 0
        all_images_train = []
        all_results_train = []
        map_metric = MeanAveragePrecision()
        for step, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            results = fcos_predictor(images, targets)
            loss = results["combined_loss"]
            loss.backward()

            if (epoch + 1) % TRAINING_MAP_EPOCHS == 0:
                map_metric = update_map(targets, results, map_metric)

            if len(all_images_train) < IMAGE_LOG_BATCHES:
                all_images_train.append(images.detach().cpu())
                all_results_train.append(results)

            if args.use_clip_grad:
                clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            losses_log += loss.detach().item()
            cls_losses_log += results["cls_loss"].detach().item()
            reg_losses_log += results["reg_loss"].detach().item()
            centerness_losses_log += results["centerness_loss"].detach().item()
            logs = {
                "loss_avg": losses_log / (step + 1),
                "cls_l": cls_losses_log / (step + 1),
                "reg_l": reg_losses_log / (step + 1),
                "cen_l": centerness_losses_log / (step + 1),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if (epoch + 1) % TRAINING_MAP_EPOCHS == 0:
            map = map_metric.compute()
            logger.log_info(pformat(map))

        if epoch % args.save_model_epochs == 0:
            model.eval()
            valid_loss = 0
            valid_cls_loss = 0
            valid_reg_loss = 0
            valid_cen_loss = 0
            with torch.no_grad():
                map_metric = MeanAveragePrecision()
                all_images_valid = []
                all_results_valid = []
                for step, (images, targets) in enumerate(
                        tqdm(eval_dataloader, total=len(eval_dataloader))):
                    images = images.to(device)

                    targets = [target.to(device) for target in targets]

                    results = fcos_predictor(images, targets)
                    loss = results["combined_loss"]
                    valid_loss += loss.detach().item()
                    valid_cls_loss += results["cls_loss"].detach().item()
                    valid_reg_loss += results["reg_loss"].detach().item()
                    valid_cen_loss += results["centerness_loss"].detach().item()

                    map_metric = update_map(targets, results, map_metric)

                    if len(all_images_valid) < IMAGE_LOG_BATCHES:
                        all_images_valid.append(images.detach().cpu())
                        all_results_valid.append(results)

                valid_loss /= len(eval_dataloader)
                valid_cls_loss /= len(eval_dataloader)
                valid_reg_loss /= len(eval_dataloader)
                valid_cen_loss /= len(eval_dataloader)

                logger.log_info(f"Epoch {epoch}")
                logger.log_info(f"Valid loss: {valid_loss}")
                logger.log_info(f"Valid cls loss loss: {valid_cls_loss}")
                logger.log_info(f"Valid reg loss: {valid_reg_loss}")
                logger.log_info(f"Valid cness loss: {valid_cen_loss}")

                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, args.output_dir)

                epoch_path = f"{logs_path}/{epoch}"
                os.makedirs(epoch_path)

                log_images(args.dataset_name, f"{epoch_path}/train",
                           all_images_train, all_results_train)
                log_images(args.dataset_name, f"{epoch_path}/valid",
                           all_images_valid, all_results_valid)

                map = map_metric.compute()
                logger.log_info(pformat(map))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--dataset_name",
                        type=str,
                        default=None,
                        help="Dataset name.")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default=None,
                        help="Dataset name.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=416)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_model_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=5e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_clip_grad", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")

    args = parser.parse_args()
    main(args)
