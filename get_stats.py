import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def read_bim_file(bim_file, target_chrom='22'):
    """
    Reads the BIM file and returns a sorted numpy array of positions for the target chromosome.
    """
    print("Reading BIM file...")
    try:
        # BIM file columns: chrom, variant, ignore1, ignore2, allele1, allele2
        bim_df = pd.read_csv(
            bim_file,
            delim_whitespace=True,
            header=None,
            usecols=[0, 1],
            names=['chrom', 'variant'],
            dtype={'chrom': str, 'variant': str}
        )
        
        # Remove 'chr' prefix from chromosome
        bim_df['chrom'] = bim_df['chrom'].str.replace('chr', '', case=False).str.strip()
        
        # Extract position from 'variant' (assuming format 'chrom:pos:allele1:allele2')
        bim_split = bim_df['variant'].str.split(':', expand=True)
        bim_df['chrom_variant'] = bim_split[0].str.replace('chr', '', case=False).str.strip()
        bim_df['pos_variant'] = bim_split[1].astype(int)
        
        # Filter to target chromosome
        bim_df = bim_df[bim_df['chrom_variant'] == target_chrom]
        
        # Extract positions and sort them
        bim_positions = np.sort(bim_df['pos_variant'].values)
        print(f"Total positions in BIM file for chr{target_chrom}: {len(bim_positions)}\n")
        return bim_positions
    except Exception as e:
        print(f"Error reading BIM file: {e}")
        return np.array([])

def read_weights_file(weights_file, target_chrom='22'):
    """
    Reads the weights CSV, filters to target_chrom, and returns a DataFrame and numpy array of positions.
    """
    print("Reading weights file...")
    try:
        # Read weights CSV with only necessary columns
        weights_df = pd.read_csv(
            weights_file,
            dtype={'chr': str, 'pos': int, 'effect_allele': str, 'weight': float, 'id': str},
            usecols=['chr', 'pos', 'effect_allele', 'weight', 'id']
        )
        
        # Remove 'chr' prefix and filter to target_chrom
        weights_df['chr'] = weights_df['chr'].str.replace('chr', '', case=False).str.strip()
        weights_df = weights_df[weights_df['chr'] == target_chrom]
        
        # Extract positions as numpy array
        weights_positions = weights_df['pos'].values
        total_weights_positions = len(weights_positions)
        print(f"Total positions in weights file (chr{target_chrom}): {total_weights_positions}\n")
        
        # Display first 5
        print("First 5 parsed variants from weights file:")
        for idx, row in weights_df.head(5).iterrows():
            print(f"  Variant {idx+1}: Chromosome: {row['chr']}, Position: {row['pos']}, "
                  f"Effect Allele: {row['effect_allele']}, Weight: {row['weight']}, ID: {row['id']}")
        print("\n")
        
        return weights_df, weights_positions
    except Exception as e:
        print(f"Error reading weights file: {e}")
        return pd.DataFrame(), np.array([])

def find_exact_matches(weights_positions, bim_positions):
    """
    Finds exact matches between weights and BIM positions.
    Returns a set of matched positions.
    """
    print("Finding exact matched positions...")
    bim_set = set(bim_positions)
    matched_positions = set(weights_positions).intersection(bim_set)
    matched_count = len(matched_positions)
    total = len(weights_positions)
    percentage = (matched_count / total) * 100 if total > 0 else 0
    print(f"Total exact matched positions: {matched_count}")
    print(f"Percentage of exact matched positions: {percentage:.2f}% out of {total} total positions in weights file\n")
    return matched_positions, matched_count, percentage

def find_approximate_matches(weights_positions, bim_positions, matched_positions, window=5000):
    """
    Finds approximate matches within ±window base pairs.
    Returns:
        - approx_matched_positions: set of positions with at least one BIM position within the window
        - distances: list of minimum distances for each approximate match
    """
    print("Finding approximate matched positions (within ±5000 bp)...")
    approx_matched_positions = set()
    distances = []
    
    # Convert BIM positions to sorted numpy array
    bim_sorted = bim_positions
    bim_sorted = np.sort(bim_sorted)
    
    for pos in tqdm(weights_positions, desc="Processing approximate matches"):
        if pos in matched_positions:
            continue  # Skip exact matches
        # Find the insertion points
        left = pos - window
        right = pos + window
        left_idx = np.searchsorted(bim_sorted, left, side='left')
        right_idx = np.searchsorted(bim_sorted, right, side='right')
        # Slice the relevant BIM positions
        relevant_bim = bim_sorted[left_idx:right_idx]
        if relevant_bim.size > 0:
            approx_matched_positions.add(pos)
            # Calculate the minimum distance
            min_distance = np.min(np.abs(relevant_bim - pos))
            distances.append(min_distance)
    
    approx_count = len(approx_matched_positions)
    total = len(weights_positions)
    approx_percentage = (approx_count / total) * 100 if total > 0 else 0
    mean_distance = np.mean(distances) if distances else 0
    print(f"\nTotal approximate matched positions (within ±5000 bp): {approx_count}")
    print(f"Percentage of approximate matched positions: {approx_percentage:.2f}% out of {total} total positions in weights file")
    print(f"Mean distance for approximate matches: {mean_distance:.2f} bp\n")
    
    return approx_matched_positions, approx_count, approx_percentage, mean_distance

def collect_examples(weights_df, matched_exact, matched_approx, example_limit=5):
    """
    Collects example exact matches, approximate matches, and non-matches.
    Returns lists of example matches and non-matches.
    """
    print("Collecting example matches and non-matches...")
    example_matches = []
    example_non_matches = []
    
    # Exact matches
    exact_matches_df = weights_df[weights_df['pos'].isin(matched_exact)]
    exact_matches_sample = exact_matches_df.head(example_limit).to_dict('records')
    for match in exact_matches_sample:
        example_matches.append({
            'type': 'Exact Match',
            'chr': match['chr'],
            'pos': match['pos'],
            'effect_allele': match['effect_allele'],
            'weight': match['weight'],
            'id': match['id']
        })
    
    # Approximate matches
    approx_matches_df = weights_df[weights_df['pos'].isin(matched_approx)]
    approx_matches_sample = approx_matches_df.head(example_limit).to_dict('records')
    for match in approx_matches_sample:
        example_matches.append({
            'type': 'Approximate Match',
            'chr': match['chr'],
            'pos': match['pos'],
            'effect_allele': match['effect_allele'],
            'weight': match['weight'],
            'id': match['id']
        })
    
    # Non-matches
    non_matches_df = weights_df[~weights_df['pos'].isin(matched_exact.union(matched_approx))]
    non_matches_sample = non_matches_df.head(example_limit).to_dict('records')
    for non_match in non_matches_sample:
        example_non_matches.append({
            'chr': non_match['chr'],
            'pos': non_match['pos'],
            'effect_allele': non_match['effect_allele'],
            'weight': non_match['weight'],
            'id': non_match['id']
        })
    
    return example_matches, example_non_matches

def main():
    weights_file = "pgs003725_processed_weights.csv"
    bim_file = "acaf_threshold.chr22.bim"
    target_chrom = '22'
    
    # Check if files exist
    if not os.path.isfile(weights_file):
        print(f"Error: Weights file '{weights_file}' does not exist.")
        return
    if not os.path.isfile(bim_file):
        print(f"Error: BIM file '{bim_file}' does not exist.")
        return
    
    # Read BIM file
    bim_positions = read_bim_file(bim_file, target_chrom=target_chrom)
    
    # Read weights file
    weights_df, weights_positions = read_weights_file(weights_file, target_chrom=target_chrom)
    
    if weights_positions.size == 0:
        print(f"No positions found in weights file for chromosome {target_chrom}. Exiting.")
        return
    
    # Find exact matches
    matched_exact, exact_count, exact_percentage = find_exact_matches(weights_positions, bim_positions)
    
    # Find approximate matches
    matched_approx, approx_count, approx_percentage, mean_distance = find_approximate_matches(
        weights_positions, bim_positions, matched_exact, window=5000
    )
    
    # Collect examples
    example_matches, example_non_matches = collect_examples(
        weights_df, matched_exact, matched_approx, example_limit=5
    )
    
    # Print results
    print("Processing complete.")
    print(f"Total exact matched positions: {exact_count}")
    print(f"Percentage of exact matched positions: {exact_percentage:.2f}% out of {len(weights_positions)} total positions in weights file\n")
    
    print(f"Total approximate matched positions (within ±5000 bp): {approx_count}")
    print(f"Percentage of approximate matched positions: {approx_percentage:.2f}% out of {len(weights_positions)} total positions in weights file")
    print(f"Mean distance for approximate matches: {mean_distance:.2f} bp\n")
    
    # Example matches
    if example_matches:
        print("Example Matches:")
        for match in example_matches:
            if match['type'] == 'Exact Match':
                match_type = "Exact Match"
            else:
                match_type = "Approximate Match"
            print(f"  {match_type}:")
            print(f"    Weights File - Chromosome: {match['chr']}, Position: {match['pos']}, "
                  f"Effect Allele: {match['effect_allele']}, Weight: {match['weight']}, ID: {match['id']}")
            print(f"    BIM File - Chromosome: {match['chr']}, Position: {match['pos']}, Variant: {match['pos']}:X:X\n")
    else:
        print("No example matches found.\n")
    
    # Example non-matches
    if example_non_matches:
        print("Example Non-Matches (from Weights File):")
        for non_match in example_non_matches:
            print(f"  Chromosome: {non_match['chr']}, Position: {non_match['pos']}, "
                  f"Effect Allele: {non_match['effect_allele']}, Weight: {non_match['weight']}, "
                  f"ID: {non_match['id']}, Match Status: No Match\n")
    else:
        print("No example non-matches found.\n")

if __name__ == "__main__":
    main()
