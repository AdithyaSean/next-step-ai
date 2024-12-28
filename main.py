"""Main entry point for the Next-Step AI application."""

import sys

from src.generators.generator import generator
from src.preprocessors.preprocessor import preprocessor
from src.train.trainer import trainer


def main():
    """Execute the main application logic based on command-line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python -m main [command]")
        print("Available commands: generate, process, train, run")
        sys.exit(1)

    command = sys.argv[1].lower()

    try:
        if command == "generate":
            print("Generating synthetic dataset...")
            generator()
        elif command == "process":
            print("Processing data...")
            preprocessor()
        elif command == "train":
            print("Training model...")
            trainer()
        elif command == "run":
            print("Running all steps...")
            print("\nStep 1: Generating synthetic dataset...")
            generator()
            print("\nStep 2: Preprocessing and Trainining...")
            trainer()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: generate, process, train, run")
            sys.exit(1)

        print("Operation completed successfully.")
    except ImportError as e:
        print(f"Error: Could not import required module. {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
