import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

RELATIVE_PATH = os.path.dirname(os.path.abspath(__file__))

REPO_DIR = os.path.dirname(RELATIVE_PATH)
TRANSFORMER_DIR = os.path.join(REPO_DIR, "transformer_owen")
sys.path.insert(0, REPO_DIR)

from transformer_owen.decoder_transformer import build_transformer
from LoRA_transformer_owen.LoRA_transformer import build_lora_transformer, LoRALayer
from flash_attention_owen.flash_attention_transformer import build_flash_attention_transformer
from transformer_owen.tokenizer import encode, get_vocab_info


class TextDataset(Dataset):
    def __init__(self, text, encoder, seq_len):
        self.seq_len = seq_len
        self.data = encoder(text)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        sequence = self.data[index : index + self.seq_len + 1]
        return (
            torch.tensor(sequence[:-1], dtype=torch.long),
            torch.tensor(sequence[1:], dtype=torch.long),
        )


def create_causal_mask(seq_len, device):
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a decoder transformer.")
    parser.add_argument(
        "--input", default=os.path.join("training_data", "shakespeare.txt")
    )
    parser.add_argument("--output-dir", default=os.path.join(RELATIVE_PATH, "outputs"))

    # these can remain default
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)

    parser.add_argument("--loss-criteria", type=float, required=False)
    parser.add_argument("--loss-patience", type=float, default=3, required=False)
    parser.add_argument("--loss-delta", type=float, default=1e-4, required=False)
    parser.add_argument("--validation-interval", type=int, default=10000)
    
    parser.add_argument("--model-type", choices=["default", "finetuning-no-lora", "lora", "flash-attention"], default="default", required=False)
    parser.add_argument("--finetuning-data", type=str, default=os.path.join("training_data", "finetuned_shakespeare.txt"), required=False)
    parser.add_argument("--finetuning-model", type=str, required=False)
    parser.add_argument("--lora-a", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=4)

    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-dblocks", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-ff", type=int, default=2048)

    return parser.parse_args()


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def main():
    args = parse_args()

    if args.iters < 1:
        raise ValueError("--train-iters must be positive")
    if args.batch_size < 1 or args.seq_length < 1:
        raise ValueError("--batch-size and --seq-length must be positive")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads")
    if args.model_type in ['lora', "finetuning-no-lora"] and args.finetuning_data is None:
        raise ValueError("--finetuning-data must be defined")
    if args.model_type in ['lora', "finetuning-no-lora"] and args.finetuning_model is None:
        raise ValueError("--finetuning-model must be defined")
    if args.model_type is not None:
        print(f"Training a {args.model_type} transformer model")
    if args.loss_criteria is not None:
        print(f"Running under loss threshold of {args.loss_criteria}")

    device = resolve_device(args.device)
    
    with open(args.input, "r", encoding="utf-8") as data_file:
        text = data_file.read()
    
    vocab_size, chars = get_vocab_info(text)
        
    if(args.model_type in ["lora", "finetuning-no-lora"]):
        with open(args.finetuning_data, "r", encoding="utf-8") as data_file:
            text = data_file.read()

    # load the dataset
    dataset = TextDataset(text, encode(chars), args.seq_length)

    validation_size = max(1, int(len(dataset) * 0.01))
    training_size = len(dataset) - validation_size

    if training_size < 1:
        raise ValueError("The input must contain at least one training example")

    training_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
    )
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    if not len(training_dataloader):
        raise ValueError("The input must contain at least one full training batch")

    # build the model
    if args.model_type in ['default', 'finetuning-no-lora']:
        model = build_transformer(
            tgt_vocab_size=vocab_size,
            tgt_seq=args.seq_length,
            d_model=args.d_model,
            N=args.num_dblocks,
            h=args.num_heads,
            dropout=args.dropout,
            d_ff=args.d_ff,
        ).to(device)
        
        if args.model_type == 'finetuning-no-lora':
            # load pretrained model for finetuning
            state_dict = torch.load(os.path.join(RELATIVE_PATH, args.finetuning_model))
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model weights")
            
    elif args.model_type == 'lora':
        model = build_lora_transformer(
            tgt_vocab_size=vocab_size, 
            tgt_seq=args.seq_length,
            d_model=args.d_model,
            N=args.num_dblocks,
            h=args.num_heads,
            dropout=args.dropout,
            d_ff=args.d_ff,
            lora_a=args.lora_a,
            lora_r=args.lora_r
        ).to(device)
        
        # load pretrained model for LoRA
        state_dict = torch.load(os.path.join(RELATIVE_PATH, args.finetuning_model))
        model.load_state_dict(state_dict, strict=False)
        print("Loaded model weights")
    
        # make sure to freeze all parameters except for the A and B matrices
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    
        for module in model.modules():
            if isinstance(module, LoRALayer):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
                
    elif args.model_type == 'flash-attention':
        model = build_flash_attention_transformer(
            tgt_vocab_size=vocab_size,
            tgt_seq=args.seq_length,
            d_model=args.d_model,
            N=args.num_dblocks,
            h=args.num_heads,
            dropout=args.dropout,
            d_ff=args.d_ff,
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.loss_criteria is not None:
        run_name = os.path.join(
            args.output_dir,
            f"{args.model_type}_seq{args.seq_length}_H{args.num_heads}_N{args.num_dblocks}_to_loss{args.loss_criteria}",
        )
    else:
        run_name = os.path.join(
            args.output_dir,
            f"{args.model_type}_seq{args.seq_length}_H{args.num_heads}_N{args.num_dblocks}_Iters{args.iters}",
        )
    os.makedirs(run_name, exist_ok=True)

    log_path = os.path.join(run_name, "log.txt")
    model_path = os.path.join(run_name, "model.pth")

    losses = []
    validation_losses = []
    validation_steps = []
    data_iterator = iter(training_dataloader)

    start_time = time.perf_counter()

    with open(log_path, "w", encoding="utf-8") as log_file:

        # log model trainable params
        params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        message = (
            f"Total Model parameters: {params:,}\n"
            f"Trainable parameters: {trainable_params:,}"
        )
        print(message)
        print(message, file=log_file)

        try:
            print(f"Running on device {device}")
            print(f"Training data loader length: {len(training_dataloader)}")
            print(f"Validation data loader length: {len(validation_dataloader)}")

            model.train()
            train_iter = 1
            best_validation_loss = None
            patience_counter = 0
            validation_counter = 0

            while train_iter <= args.iters or args.loss_criteria is not None:
                try:
                    x, y = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(training_dataloader)
                    x, y = next(data_iterator)

                x = x.to(device)
                y = y.to(device)
                mask = create_causal_mask(args.seq_length, device)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(x, mask.unsqueeze(0))
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[-1]),
                    y.reshape(-1),
                )
                loss.backward()
                optimizer.step()

                # save checkpoint on interval
                if (args.checkpoint_interval and train_iter % args.checkpoint_interval == 0):
                    checkpoint_path = os.path.join(
                        args.output_dir,
                        f"{run_name}_iter{train_iter}.pth",
                    )
                    torch.save(model.state_dict(), checkpoint_path)

                # logging results
                losses.append(loss.item())
                if train_iter % args.log_interval == 0 or train_iter == 1:
                    message = (
                        f"Iteration {train_iter}/{args.iters} "
                        f"\ttraining loss: {loss.item():.6f} "
                        f"\telapsed time: {(time.perf_counter() - start_time):.6f} secs "
                    )
                    print(message)
                    print(message, file=log_file)
                    
                check_condition = args.loss_criteria
                if check_condition is None:
                    check_condition = 0

                # check validation after set interval or training loss is below threshold
                if validation_counter >= args.validation_interval or loss.item() < check_condition:
                    model.eval()
                    validation_counter = 0
                    validation_loss_total = 0.0
                    validation_token_count = 0
                    with torch.no_grad():
                        for validation_x, validation_y in validation_dataloader:
                            validation_x = validation_x.to(device)
                            validation_y = validation_y.to(device)
                            validation_outputs = model(
                                validation_x,
                                mask.unsqueeze(0),
                            )
                            validation_loss = criterion(
                                validation_outputs.reshape(
                                    -1, validation_outputs.shape[-1]
                                ),
                                validation_y.reshape(-1),
                            )
                            validation_loss_total += (
                                validation_loss.item() * validation_y.numel()
                            )
                            validation_token_count += validation_y.numel()
                            
                    model.train()
                    validation_loss_value = (
                        validation_loss_total / validation_token_count
                    )
                    validation_losses.append(validation_loss_value)
                    validation_steps.append(train_iter)

                    message = f"validation loss: {validation_loss_value:.6f}"
                    print(message)
                    print(message, file=log_file)

                    # stopping criteria for loss if very little improvement
                    if best_validation_loss is None:
                        best_validation_loss = validation_loss_value
                    elif validation_loss_value > best_validation_loss - args.loss_delta:
                        patience_counter += 1
                        if patience_counter >= args.loss_patience:
                            message = "Validation loss not improving: stopping training"
                            print(message)
                            print(message, file=log_file)
                            break
                    else:
                        best_validation_loss = validation_loss_value
                        patience_counter = 0

                    # stop if loss meets criteria
                    if (
                        args.loss_criteria is not None
                        and validation_loss_value <= args.loss_criteria
                    ):
                        break

                validation_counter += 1
                train_iter += 1

        except KeyboardInterrupt:
            message = "Interrupt handle, saving results"
            print(message)
            print(message, file=log_file)

        finally:
            elapsed = time.perf_counter() - start_time
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            message = (
                f"Final training time: {elapsed:.6f} secs\n"
                f"Peak GPU memory: {peak_mem:.6f} MB "
            )
            print(message, file=log_file)
            print(message)

            torch.save(model.state_dict(), model_path)

            plt.plot(
                range(1, len(losses) + 1), losses, label="Training Loss", color="blue"
            )
            plt.plot(
                validation_steps,
                validation_losses,
                label="Validation Loss",
                color="red",
                linewidth=2,
            )
            plt.xlabel("Training iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(run_name, "loss_graph.png"))
            plt.close()

            print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
