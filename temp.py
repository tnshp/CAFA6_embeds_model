from Dataset.Utils import read_fasta

#break into chunck of 100k swequences and save to file 
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))    

sequences = read_fasta("/mnt/d/ML/Kaggle/CAFA6/cafa-6-protein-function-prediction/Test/testsuperset.fasta")
entrys = list(sequences.keys())
print(f"Total sequences: {len(sequences)}")
print(f"First 5 sequences: {entrys[:5]}")

#save the entrys into chuncks of 100k
chunk_size = 100000
for i, chunk in enumerate(chunker(entrys, chunk_size)):
    with open(f"/mnt/d/ML/Kaggle/CAFA6/cafa-6-protein-function-prediction/Test/test_entrys_chunk_{i}.txt", "w") as f:
        for entry in chunk:
            f.write(f"{entry} ")
    print(f"Saved chunk {i} with {len(chunk)} entrys")