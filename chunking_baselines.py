import os
from data.preprocess import load_msmarco_subset
from chunking.passage_native import PassageNativeChunker
from chunking.fixed_size import FixedSizeChunker

def main():
    print("Loading a small sample of 5 queries from MSMARCO-XI...")
    passages = load_msmarco_subset(num_queries=5)
    
    # We will pick just 5 passages for the side-by-side comparison to keep output manageable
    sample_passages = passages[:5]
    
    print(f"Loaded {len(sample_passages)} sample passages.\n")
    
    pn_chunker = PassageNativeChunker()
    fs_chunker = FixedSizeChunker(chunk_size=50, overlap_percent=0.20) 
    # Note: using 50 chunk_size for the sample visualization so we can actually see splits if passages are long.
    
    pn_chunks = pn_chunker.chunk(sample_passages)
    fs_chunks = fs_chunker.chunk(sample_passages)
    
    # Write to data/chunking_sample.md
    output_path = "data/chunking_sample.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Chunking Baseline Comparison\n\n")
        
        for p in sample_passages:
            f.write(f"## Source Passage ID: {p.passage_id} ({p.language})\n")
            f.write(f"**Original Text Length**: {len(p.text.split())} words\n\n")
            
            p_pn_chunks = [c for c in pn_chunks if c.source_passage_id == p.passage_id]
            f.write(f"### Passage-Native Chunker (Total Chunks: {len(p_pn_chunks)})\n")
            for c in p_pn_chunks:
                f.write(f"- **Chunk ID**: `{c.chunk_id}`\n")
                f.write(f"- **Text**: {c.text}\n\n")
                
            p_fs_chunks = [c for c in fs_chunks if c.source_passage_id == p.passage_id]
            f.write(f"### Fixed-Size Chunker (Total Chunks: {len(p_fs_chunks)})\n")
            f.write("*Note: Configured with 50 tokens/20% overlap for visualization*\n\n")
            for c in p_fs_chunks:
                f.write(f"- **Chunk ID**: `{c.chunk_id}`\n")
                f.write(f"- **Text**: {c.text}\n\n")
                
            f.write("---\n\n")
            
    print(f"Successfully wrote comparison to {output_path}")

if __name__ == "__main__":
    main()
