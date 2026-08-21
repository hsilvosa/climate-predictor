"""Unified Command Line Interface for Climate Forecast system."""

from __future__ import annotations

import argparse
import sys
import uvicorn

from climate_forecast.config import (
    CHECKPOINTS_DIR,
    PROCESSED_DATA_DIR,
    ModelConfig,
    TrainingConfig,
)


def run_prepare_data(args):
    from scripts.01_prepare_dataset import main as prep_main
    sys.argv = ["01_prepare_dataset.py", "--num-stations", str(args.num_stations)]
    prep_main()


def run_train(args):
    from scripts.02_train_model import main as train_main
    sys.argv = [
        "02_train_model.py",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
    ]
    train_main()


def run_evaluate(args):
    from scripts.03_evaluate_benchmark import main as eval_main
    sys.argv = ["03_evaluate_benchmark.py"]
    eval_main()


def run_export_hf(args):
    from scripts.04_export_to_hf import main as hf_main
    sys.argv = ["04_export_to_hf.py"]
    hf_main()


def run_serve(args):
    uvicorn.run(
        "climate_forecast.web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="climate-forecast",
        description="Global Extreme Climate & Heatwave Forecaster CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # prepare-data
    p_prep = subparsers.add_parser("prepare-data", help="Curate NOAA GHCN dataset & 30-year climatology")
    p_prep.add_argument("--num-stations", type=int, default=120)

    # train
    p_train = subparsers.add_parser("train", help="Train SpatioTemporal Neural Forecaster")
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate benchmark skill against baselines")

    # export-hf
    p_exp = subparsers.add_parser("export-hf", help="Export safetensors, ONNX, and Model Card for Hugging Face")

    # serve
    p_serve = subparsers.add_parser("serve", help="Launch FastAPI web dashboard and REST server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "prepare-data":
        run_prepare_data(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "export-hf":
        run_export_hf(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
