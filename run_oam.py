"""
Entry point for OAM dataset applications.

Usage:

Pixel DDPM:
    python run_oam.py --mode train_ddpm     --mat_path /path/data.mat
    python run_oam.py --mode sample_ddpm    --resume checkpoints_ddpm_oam/ckpt_300000.pt
    python run_oam.py --mode progression_ddpm --resume checkpoints_ddpm_oam/ckpt_300000.pt

VAE:
    python run_oam.py --mode train_vae      --mat_path /path/data.mat --image_size 128
    python run_oam.py --mode visualize_vae  --vae_checkpoint checkpoints_vae_oam/vae_oam_epoch100.pt --mat_path /path/data.mat
    python run_oam.py --mode eval_vae_quality --vae_checkpoint checkpoints_vae_oam/vae_oam_epoch100.pt --mat_path /path/data.mat

Latent DDPM:
    Pre-requisite: retrain VAE at 128x128 (see train_vae mode above) and use that checkpoint here.
    python run_oam.py --mode train_ldm      --vae_checkpoint checkpoints_vae_oam/vae_oam_epoch100.pt --mat_path /path/data.mat
    python run_oam.py --mode sample_ldm     --ldm_checkpoint checkpoints_ldm/ldm_ckpt_200000.pt --vae_checkpoint checkpoints_vae_oam/vae_oam_epoch100.pt

CNN turbulence classifier:
    Train cnn on real laser images, then evaluate on generated samples.
    python run_oam.py --mode train_cnn      --mat_path /path/data.mat
    python run_oam.py --mode eval_cnn       --resume checkpoints_cnn/best_cnn.pt --output_dir samples_ddpm

Outside of scope: specific analysis scripts, like interpolation (use analyse_interp_latent.py for that)
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="OAM DDPM, VAE, VAE+DDPM pipeline")

    parser.add_argument("--mode", required=True,
                        choices=[
                            # Pixel DDPM
                            "train_ddpm", "sample_ddpm", "progression_ddpm",
                            # VAE
                            "train_vae", "visualize_vae", "eval_vae_quality",
                            # Latent DDPM
                            "train_ldm", "sample_ldm",
                            # CNN turbulence classifier
                            "train_cnn", "eval_cnn",
                        ])

    # Data
    parser.add_argument("--mat_path",    type=str, default=None)
    parser.add_argument("--modes",       type=str, nargs="+", default=None,
                        help="OAM modes to include, e.g. --modes gauss p1 p4")
    parser.add_argument("--turb_levels", type=int, nargs="+", default=None,
                        help="Turbulence levels to include, e.g. --turb_levels 1 2 3")

    # Checkpoints
    parser.add_argument("--resume",          type=str, default=None,
                        help="Checkpoint to resume training or use for sampling/eval")
    parser.add_argument("--vae_checkpoint",  type=str, default=None)
    parser.add_argument("--ldm_checkpoint",  type=str, default=None)
    parser.add_argument("--pixel_checkpoint", type=str, default=None,
                        help="Pixel DDPM checkpoint")

    # Training
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--total_steps", type=int,   default=300_000)
    parser.add_argument("--total_epochs", type=int,  default=100)
    parser.add_argument("--kl_weight",   type=float, default=1e-4)
    parser.add_argument("--save_dir",    type=str,   default=None,
                        help="Checkpoint save directory. Defaults per mode if omitted.")
    parser.add_argument("--save_every",  type=int,   default=50_000)
    parser.add_argument("--log_every",   type=int,   default=1000)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--subset_size", type=int,   default=None)

    # Model / data shape
    parser.add_argument("--image_size",         type=int,  default=128)
    parser.add_argument("--vae_channel_mults",  type=int,  nargs="+", default=None)

    # Sampling / eval
    parser.add_argument("--n_samples",   type=int, default=64)
    parser.add_argument("--n_eval",      type=int, default=None)
    parser.add_argument("--n_frames",    type=int, default=10)
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Output directory. Defaults per mode if omitted.")

    # Misc
    parser.add_argument("--device",   type=str, default="cuda")
    parser.add_argument("--no_tsne",  action="store_true")
    parser.add_argument("--n_per_cell", type=int, default=8,
                        help="Images per cell in VAE quality grid")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Pixel DDPM
    # ------------------------------------------------------------------

    if args.mode == "train_ddpm":
        if args.mat_path is None:
            parser.error("--mat_path is required for train_ddpm")
        from train_ddpm_oam import train
        train(
            mat_path=args.mat_path,
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps,
            save_dir=args.save_dir or "checkpoints_ddpm_oam",
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            image_size=args.image_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size,
            turb_levels=args.turb_levels,
            modes=args.modes,
        )

    elif args.mode == "sample_ddpm":
        if args.resume is None:
            parser.error("--resume is required for sample_ddpm")
        from sample_oam import sample
        sample(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            output_dir=args.output_dir or "samples_ddpm_oam",
            device=args.device,
            image_size=args.image_size,
        )

    elif args.mode == "progression_ddpm":
        if args.resume is None:
            parser.error("--resume is required for progression_ddpm")
        from sample_oam import sample_progression
        sample_progression(
            checkpoint_path=args.resume,
            n_samples=args.n_samples,
            n_frames=args.n_frames,
            output_dir=args.output_dir or "samples_ddpm_oam",
            device=args.device,
            image_size=args.image_size,
        )

    # ------------------------------------------------------------------
    # VAE
    # ------------------------------------------------------------------

    elif args.mode == "train_vae":
        if args.mat_path is None:
            parser.error("--mat_path is required for train_vae")
        from train_vae_oam import train_vae_oam
        train_vae_oam(
            mat_path=args.mat_path,
            batch_size=args.batch_size,
            lr=args.lr,
            total_epochs=args.total_epochs,
            kl_weight=args.kl_weight,
            save_dir=args.save_dir or "checkpoints_vae_oam",
            save_every=10,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            num_workers=args.num_workers,
            image_size=args.image_size,
            channel_mults=args.vae_channel_mults,
            modes=args.modes,
            turb_levels=args.turb_levels,
        )

    elif args.mode == "visualize_vae":
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for visualize_vae")
        if args.mat_path is None:
            parser.error("--mat_path is required for visualize_vae")
        from visualize_latent import visualize_oam
        visualize_oam(
            vae_checkpoint=args.vae_checkpoint,
            mat_path=args.mat_path,
            output_dir=args.output_dir or "vis_vae_oam",
            device=args.device,
            tsne=not args.no_tsne,
            pca_scatter=False,
            interpolation=True,
            traversal=False,
            reconstruction=False,
        )

    elif args.mode == "eval_vae_quality":
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for eval_vae_quality")
        if args.mat_path is None:
            parser.error("--mat_path is required for eval_vae_quality")
        from analyse_vae_quality import main as vae_quality
        vae_quality(
            vae_checkpoint=args.vae_checkpoint,
            mat_path=args.mat_path,
            output_dir=args.output_dir or "analysis_vae_quality",
            n_per_cell=args.n_per_cell,
            device=args.device,
            modes=args.modes,
            turb_levels=args.turb_levels,
        )

    # ------------------------------------------------------------------
    # Latent DDPM
    # ------------------------------------------------------------------

    elif args.mode == "train_ldm":
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for train_ldm")
        if args.mat_path is None:
            parser.error("--mat_path is required for train_ldm")
        from train_ddpm_latent import train
        train(
            vae_checkpoint=args.vae_checkpoint,
            mat_path=args.mat_path,
            batch_size=args.batch_size,
            lr=args.lr,
            total_steps=args.total_steps,
            save_dir=args.save_dir or "checkpoints_ldm",
            save_every=args.save_every,
            log_every=args.log_every,
            resume=args.resume,
            device=args.device,
            num_workers=args.num_workers,
            modes=args.modes,
            turb_levels=args.turb_levels,
        )

    elif args.mode == "sample_ldm":
        if args.ldm_checkpoint is None:
            parser.error("--ldm_checkpoint is required for sample_ldm")
        if args.vae_checkpoint is None:
            parser.error("--vae_checkpoint is required for sample_ldm")
        from sample_ldm import main as sample_ldm_main
        sample_ldm_main(
            ldm_checkpoint=args.ldm_checkpoint,
            vae_checkpoint=args.vae_checkpoint,
            output_dir=args.output_dir or "samples_ldm",
            n_samples=args.n_samples,
            pixel_checkpoint=args.pixel_checkpoint,
            image_size=args.image_size,
            device=args.device,
        )


    # ------------------------------------------------------------------
    # CNN turbulence classifier
    # ------------------------------------------------------------------

    elif args.mode == "train_cnn":
        if args.mat_path is None:
            parser.error("--mat_path is required for train_cnn")
        import types
        from cnn_turb_classifier import train as train_cnn
        cnn_args = types.SimpleNamespace(
            mat_path=args.mat_path,
            save_dir=args.save_dir or "checkpoints_cnn",
            epochs=args.total_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=3,
            turb_levels=args.turb_levels,
            modes=args.modes,
            num_workers=args.num_workers,
        )
        train_cnn(cnn_args)

    elif args.mode == "eval_cnn":
        if args.resume is None:
            parser.error("--resume is required for eval_cnn")
        if args.output_dir is None:
            parser.error("--output_dir (directory of PNG samples) is required for eval_cnn")
        import types
        from cnn_turb_classifier import evaluate_ddpm
        cnn_args = types.SimpleNamespace(
            checkpoint=args.resume,
            eval_dir=args.output_dir,
        )
        evaluate_ddpm(cnn_args)


if __name__ == "__main__":
    main()
