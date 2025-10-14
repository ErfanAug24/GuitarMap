from core.pipeline import GuitarAnalysisPipeline

pipeline = GuitarAnalysisPipeline()
results = pipeline.run("C:\\Users\\jill\\Downloads\\Carnival of Rust - Instrumental.mp3",
                       export_name="EnterSandman_tab")
print(f"JSON: {results['json']}")
print(f"CSV: {results['csv']}")
print(f"First 5 mapped notes:")
for note in results['tabs'][:5]:
    print(note)
