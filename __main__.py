"""Main entry point for the Next-Step AI application."""

import sys


def main():
    """Execute the main application logic based on command-line arguments."""
    if len(sys.argv) < 2:
        print("Please provide a command.")
        print("Available commands: generate, process, train, run")
        sys.exit(1)

    command = sys.argv[1]

    try:
        from src.data.preprocessors.preprocessor import preprocess_data
        from src.generators.generator import generate_synthetic_data
        from src.models.train.trainer import train_model

        if command == "generate":
            print("Generating synthetic dataset...")
            generate_synthetic_data()
        elif command == "process":
            print("Processing data...")
            preprocess_data()
        elif command == "train":
            print("Training model...")
            train_model()
        elif command == "run":
            print("Running all steps...")
            print("\nStep 1: Generating synthetic dataset...")
            generate_synthetic_data()
            print("\nStep 2: Processing data...")
            preprocess_data()
            print("\nStep 3: Training model...")
            train_model()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: generate, process, train, run")
            sys.exit(1)

    except ImportError as e:
        print(f"Error: Could not import required module. {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
