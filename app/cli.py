import argparse
from app.core.pipeline import GuitarAnalysisPipeline

def main():
    parser = argparse.ArgumentParser(description="Guitar tab extraction from audio")
    parser.add_argument("--input", "-i", required=True, help="Path to audio file")
    parser.add_argument("--export", "-e", default=None, help="Base name for exports")
    parser.add_argument("--sr", type=int, default=44100, help="Sample rate")
    args = parser.parse_args()

    pipeline = GuitarAnalysisPipeline(sr=args.sr)
    result = pipeline.run(args.input, export_name=args.export)
    print("Pipeline finished. Exports:")
    print(result)

if __name__ == "__main__":
    main()
