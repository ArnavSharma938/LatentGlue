import os
import sys
import warnings
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
from pybloom_live import BloomFilter
from rdkit import RDLogger
import gc
import json
import pickle

warnings.filterwarnings("ignore", category=RuntimeWarning, message="to-Python converter.*")
RDLogger.DisableLog('rdApp.*')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCREENING_DIR = os.path.join(ROOT_DIR, "data", "screening")
sys.path.append(ROOT_DIR)
from src.casestudy.filter import process_molecule, ATOM_PATTERN

def main():
    os.chdir(ROOT_DIR)
    input_path = os.path.join(SCREENING_DIR, "2025.02_Enamine_REAL_DB_104M.cxsmiles")
    temp_output_path = os.path.join(SCREENING_DIR, "2025.02_Enamine_REAL_intermediate.csv")
    final_output_path = os.path.join(SCREENING_DIR, "GlueDegradDB-Filter.csv")
    
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    
    num_workers = max(1, mp.cpu_count() - 1)
    chunk_size = 1_000_000
    batch_size = 250
    
    bloom = BloomFilter(capacity=150_000_000, error_rate=0.000001)
    
    stats = {"total": 0, "pre_filtered": 0, "rdkit_passed": 0, "bloom_dropped": 0}
    cols_to_read = ['smiles', 'id', 'MW', 'HBA', 'HBD', 'RotBonds', 'FSP3', 'TPSA', 'sLogP']
    checkpoint_dir = os.path.join(SCREENING_DIR, "checkpoints")
    checkpoint_meta = os.path.join(checkpoint_dir, "meta.json")
    checkpoint_bloom = os.path.join(checkpoint_dir, "bloom.pkl")
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_chunk = 1
    
    if os.path.exists(checkpoint_meta) and os.path.exists(checkpoint_bloom):
        try:
            with open(checkpoint_meta, 'r') as f:
                ckpt_data = json.load(f)
                start_chunk = ckpt_data['chunk_num'] + 1
                stats = ckpt_data['stats']
            with open(checkpoint_bloom, 'rb') as f:
                bloom = pickle.load(f)
            print(f"Resuming from Chunk {start_chunk}...")
        except Exception as e:
            print(f"Failed to load checkpoint, starting fresh: {e}")

    print(f"Workers: {num_workers} | Vectorized Chunk Size: {chunk_size}")
    mode = 'a' if start_chunk > 1 else 'w'
    with open(temp_output_path, mode, encoding='utf-8') as f_out:
        if mode == 'w':
            f_out.write("id,smiles,mw,hbd,hba,rot_bonds,net_charge,ring_count,fsp3,tpsa,aromatic_rings,total_stereo,undefined_stereo,logp\n")
        
        
        with mp.Pool(num_workers, maxtasksperchild=8000) as pool:
            reader = pd.read_csv(input_path, sep='\t', chunksize=chunk_size, 
                                low_memory=False, usecols=cols_to_read, engine='c')
            
            chunk_num = 1
            for chunk in reader:
                if chunk_num < start_chunk:
                    chunk_num += 1
                    continue

                stats["total"] += len(chunk)
                print(f"\n[GLOBAL] Handled: {stats['total']/1e6:.1f}M / 104M | Filtered: {stats['rdkit_passed']}")

                chunk = chunk[~chunk['smiles'].str.contains('.', regex=False)]
                mask = (
                    (chunk['MW'] >= 250) & (chunk['MW'] <= 600) &
                    (chunk['HBA'] <= 9) &
                    (chunk['HBD'] <= 5) &
                    (chunk['RotBonds'] <= 8) &
                    (chunk['FSP3'] >= 0.2) &
                    (chunk['TPSA'] >= 40) & (chunk['TPSA'] <= 120) &
                    (chunk['sLogP'] > 0) & (chunk['sLogP'] < 5)
                )
                survivors = chunk[mask]
                
                if not survivors.empty:
                    survivors = survivors[survivors['smiles'].str.match(ATOM_PATTERN.pattern, na=False)]
                
                if survivors.empty:
                    chunk_num += 1
                    continue
                
                stats["pre_filtered"] += len(survivors)
                
                pool_data = zip(
                    survivors['smiles'].values,
                    survivors['id'].values,
                    survivors[['MW', 'HBA', 'HBD', 'RotBonds', 'FSP3', 'TPSA', 'sLogP']].values
                )

                desc = f"Chunk {chunk_num}"
                write_buffer = []

                with tqdm(total=len(survivors), desc=desc, unit="mol", ncols=100, leave=True) as pbar:
                    update_buffer = 0
                    for result in pool.imap_unordered(process_molecule, pool_data, chunksize=batch_size):
                        if result:
                            mol_id, canonical_smiles, csv_line, pass_filter = result
                            
                            if pass_filter:
                                if canonical_smiles not in bloom:
                                    try:
                                        bloom.add(canonical_smiles)
                                    except IndexError:
                                        """
                                        I set Bloom to 30M in my run so it reached capacity, 
                                        so I had to implement this but any user can remove
                                        it now that Bloom is set to 150M
                                        """
                                        pass 
                                    stats["rdkit_passed"] += 1
                                    write_buffer.append(csv_line)
                                else:
                                    stats["bloom_dropped"] += 1
                        
                        update_buffer += 1
                        if update_buffer >= 1000:
                            pbar.update(update_buffer)
                            update_buffer = 0
                            pbar.set_postfix(passed=stats["rdkit_passed"], drop=stats["bloom_dropped"])
                    
                    if update_buffer > 0:
                        pbar.update(update_buffer)
                
                if write_buffer:
                    f_out.write("".join(write_buffer))
                    write_buffer = []

                del pool_data
                del survivors
                del chunk
                gc.collect()
                
                with open(checkpoint_meta, 'w') as f:
                    json.dump({'chunk_num': chunk_num, 'stats': stats}, f)
                with open(checkpoint_bloom, 'wb') as f:
                    pickle.dump(bloom, f)
                
                chunk_num += 1

    print(f"\n\nPre-Filtered: {stats['pre_filtered']} | RDKit Passed: {stats['rdkit_passed']}")
    
    df_final = pd.read_csv(temp_output_path)
    df_final.drop_duplicates(subset=['smiles'], keep='first', inplace=True)
    
    print(f"Finalizing {len(df_final)} unique molecules...")
    df_final.to_csv(final_output_path, index=False)
    
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
    
    if os.path.exists(checkpoint_meta):
        os.remove(checkpoint_meta)
    if os.path.exists(checkpoint_bloom):
        os.remove(checkpoint_bloom)
    if os.path.exists(checkpoint_dir):
        try:
            os.rmdir(checkpoint_dir)
        except:
            pass

if __name__ == "__main__":
    main()
