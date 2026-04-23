import argparse

from mlops_pipeline.pipeline import train_and_register_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register the article tagger model")
    parser.add_argument("--data", required=True, help="Path to medium_articles.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--model-dir", default="./distilbert_tagging_model")
    args = parser.parse_args()

    summary = train_and_register_model(
        csv_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_dir=args.model_dir,
    )
    print(summary)
