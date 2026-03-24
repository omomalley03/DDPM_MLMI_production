"""
Entry point for CIFAR-10 DDPM training and evaluation.

Usage:
    python run_cifar.py --mode train --total_steps 1300000
    python run_cifar.py --mode sample --resume checkpoints/ckpt_1300000.pt
    python run_cifar.py --mode denoise --resume checkpoints/ckpt_1300000.pt
    python run_cifar.py --mode eval   --resume checkpoints/ckpt_1300000.pt
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 DDPM")

    parser.add_argument("--mode", required=True,
                        choices=["train", "sample", "denoise", "eval"])

    # Training
    parser.add_argument("--batch_size",   type=int,   default=128)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--total_steps",  type=int,   default=1_300_000)
    parser.add_argument("--save_dir",     type=str,   default="checkpoints")
    parser.add_argument("--save_every",   type=int,   default=50_000)
    parser.add_argument("--log_every",    type=int,   default=1000)
    parser.add_argument("--resume",       type=str,   default=None)
    parser.add_argument("--device",       type=str,   default="cuda")
    parser.add_argument("--image_size",   type=int,   default=32)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--subset_size",  type=int,   default=None)

    # Sampling / eval
    parser.add_argument("--n_samples",  type=int, default=64)
    parser.add_argument("--n_eval",     type=int, default=50_000)
    parser.add_argument("--n_frames",   type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="samples")

    args = parser.parse_args()

    if args.mode == "train":
        from train import train
        train(
            dataset="cifar10",
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps,
            save_dir=args.save_dir,
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            image_size=args.image_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size,
        )

    elif args.mode == "sample":
        if args.resume is None:
            parser.error("--resume is required for sample mode")
        from sample import sample
        sample(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )

    elif args.mode == "denoise":
        if args.resume is None:
            parser.error("--resume is required for denoise mode")
        from sample import sample_progression
        sample_progression(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            n_frames=args.n_frames,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )

    elif args.mode == "eval":
        if args.resume is None:
            parser.error("--resume is required for eval mode")
        from eval import evaluate
        evaluate(
            checkpoint_path=args.resume,
            n_eval=args.n_eval,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            device=args.device,
            image_size=args.image_size,
        )


if __name__ == "__main__":
    main()
