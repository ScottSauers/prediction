#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import subprocess
import requests
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # In case display not available
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import scipy.stats as stats
import hail as hl
import gcsfs

#####################################
# Utility Functions
#####################################

def hail_already_initialized():
    try:
        _ = hl.current_backend()
        return True
    except ValueError:
        return False

def load_or_compute(cache_path, compute_func):
    print(f"Checking if {cache_path} exists for caching...")
    if hl.utils.hadoop_exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        # Decide if .mt or .ht
        if cache_path.endswith(".mt"):
            return hl.read_matrix_table(cache_path)
        elif cache_path.endswith(".ht"):
            return hl.read_table(cache_path)
        else:
            raise ValueError("Cache file must end with .mt or .ht")
    else:
        print(f"Cache not found at: {cache_path}, computing now...")
        result = compute_func()
        print(f"Writing newly computed result to cache: {cache_path}")
        if isinstance(result, hl.MatrixTable):
            result.write(cache_path, overwrite=True)
        elif isinstance(result, hl.Table):
            result.write(cache_path, overwrite=True)
        else:
            raise ValueError("Unexpected result type: must be Hail Table or MatrixTable.")
        return result

def safe_get_env(varname, required=True):
    val = os.getenv(varname)
    if val is None and required:
        raise EnvironmentError(f"Missing required environment variable: {varname}")
    return val

#####################################
# Start Script
#####################################
start_time = datetime.datetime.now()
print("Script start date:", start_time.strftime("%Y-%m-%d"))
print("Script start time:", start_time.strftime("%H:%M:%S"))

bucket = safe_get_env("WORKSPACE_BUCKET", required=True)
print("Workspace Bucket:", bucket)

cache_dir = "cache"
if not os.path.exists(cache_dir):
    print("Creating local cache directory:", cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
else:
    print("Local cache directory already exists:", cache_dir)

# Initialize Hail if not already
if not hail_already_initialized():
    print("Initializing Hail with default_reference='GRCh38'...")
    hl.init(tmp_dir=f'{bucket}/hail_temp/', default_reference='GRCh38')
    print("Hail initialized successfully.")
else:
    print("Hail is already initialized, skipping hl.init()")

#####################################
# Load and Filter VDS by Flagged Samples
#####################################
vds_path = os.getenv("WGS_VDS_PATH")
if vds_path is None:
    raise EnvironmentError("WGS_VDS_PATH not set. Cannot load VDS.")
print(f"Loading original VDS from {vds_path}...")

try:
    vds = hl.vds.read_vds(vds_path)
except Exception as e:
    raise RuntimeError(f"Failed to read VDS from {vds_path}: {e}")

num_samples = vds.n_samples()
print("Initial sample count in VDS:", num_samples)
if num_samples == 0:
    print("No samples in VDS. Exiting early.")
    exit(0)

# Print locus type
print("VDS locus dtype:", vds.variant_data.locus.dtype)

one_variant = vds.variant_data.rows().take(1)
if len(one_variant) == 0:
    print("No variants in VDS. Exiting early.")
    exit(0)
print("One variant from VDS:", one_variant)
locus_example = one_variant[0].locus
contig_example = locus_example.contig
print("This variant's contig naming is:", contig_example)

# Filter flagged samples
flagged_samples_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"
print(f"Importing flagged samples from {flagged_samples_path}...")
try:
    flagged_ht = hl.import_table(flagged_samples_path, key='sample_id')
except Exception as e:
    raise RuntimeError(f"Failed to import flagged samples: {e}")

print("Flagged samples table schema:")
flagged_ht.describe()

def filter_flagged_samples_func():
    print("Filtering flagged samples from VDS...")
    return hl.vds.filter_samples(vds, flagged_ht, keep=False)

vds_no_flag = filter_flagged_samples_func()
num_after_flag = vds_no_flag.n_samples()
print("Samples after removing flagged:", num_after_flag)
if num_after_flag == 0:
    print("All samples flagged. No samples remain. Exiting.")
    exit(0)

#####################################
# Filter by WGS+EHR Samples
#####################################
print("Querying BigQuery for WGS+EHR samples...")
workspace_cdr = os.getenv("WORKSPACE_CDR")
if workspace_cdr is None:
    print("WORKSPACE_CDR not set. Cannot run BigQuery query. Exiting.")
    exit(0)

query = f"""
SELECT person_id
FROM `{workspace_cdr}.person`
WHERE person_id IN (
    SELECT DISTINCT person_id
    FROM `{workspace_cdr}.cb_search_person`
    WHERE has_ehr_data = 1 AND has_whole_genome_variant = 1
)
"""
print("Running query:\n", query)

try:
    sample_ids_df = pd.read_gbq(query, dialect='standard')
except Exception as e:
    print(f"Failed to run BigQuery query: {e}")
    exit(0)

count_wgs_ehr = sample_ids_df['person_id'].nunique()
print("Number of samples with WGS & EHR data:", count_wgs_ehr)
if count_wgs_ehr == 0:
    print("No samples with WGS & EHR data. Exiting.")
    exit(0)

sample_ids_path = f'{bucket}/prs_calculator_tutorial/people_with_WGS_EHR_ids.csv'
if not hl.utils.hadoop_exists(sample_ids_path):
    print(f"Saving sample IDs to {sample_ids_path}...")
    try:
        sample_ids_df.to_csv(sample_ids_path, index=False)
    except Exception as e:
        print(f"Failed to save sample IDs: {e}")
        exit(0)
else:
    print(f"Sample IDs file already exists at {sample_ids_path}, skipping write.")

print("Importing sample IDs into Hail Table...")
try:
    sample_ids_ht = hl.import_table(sample_ids_path, delimiter=',', key='person_id')
except Exception as e:
    print(f"Failed to import sample IDs table: {e}")
    exit(0)

print("Sample IDs table schema:")
sample_ids_ht.describe()

def filter_wgs_ehr():
    print("Filtering VDS for WGS+EHR samples...")
    return hl.vds.filter_samples(vds_no_flag, sample_ids_ht, keep=True)

vds_filtered = filter_wgs_ehr()
num_after_wgs_ehr = vds_filtered.n_samples()
print("Samples after WGS+EHR filtering:", num_after_wgs_ehr)
if num_after_wgs_ehr == 0:
    print("No samples remain after WGS+EHR filtering. Exiting.")
    exit(0)

#####################################
# Prepare PRS Weight Table (Interval Filtering)
#####################################
prs_identifier = 'PGS003446'
pgs_weight_url_harmonized = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003446/ScoringFiles/Harmonized/PGS003446_hmPOS_GRCh38.txt.gz'
harmonized_local_gz = 'PGS003446_hmPOS_GRCh38.txt.gz'
harmonized_local_txt = 'PGS003446_hmPOS_GRCh38.txt'

if not os.path.exists(harmonized_local_txt):
    print("Downloading harmonized PRS weight file...")
    try:
        resp = requests.get(pgs_weight_url_harmonized, timeout=120)
        print("Response status code for PRS file download:", resp.status_code)
        if resp.status_code != 200:
            print("Failed to download PRS weight file. Status:", resp.status_code)
            exit(0)
        with open(harmonized_local_gz, 'wb') as f:
            f.write(resp.content)
        print("Download completed, now decompressing...")
        subprocess.run(["gunzip", "-f", harmonized_local_gz], check=True)
    except Exception as e:
        print(f"Failed to download or decompress PRS weight file: {e}")
        exit(0)
else:
    print("Harmonized PRS weight file already exists locally, skipping download.")

if not os.path.exists(harmonized_local_txt):
    print("Harmonized PRS file still not found. Exiting.")
    exit(0)

print(f"Reading PRS weights from {harmonized_local_txt}...")
try:
    prs_df = pd.read_csv(harmonized_local_txt, sep='\t', comment='#')
except Exception as e:
    print(f"Failed to read PRS file: {e}")
    exit(0)

num_prs_variants = len(prs_df)
print("Number of variants in PRS weight file:", num_prs_variants)
if num_prs_variants == 0:
    print("No variants in PRS weight file. Exiting.")
    exit(0)

print("Renaming columns and checking contig format in PRS...")
required_cols = ['hm_chr','hm_pos','effect_allele','other_allele','effect_weight']
for c in required_cols:
    if c not in prs_df.columns:
        print(f"Missing required column {c} in PRS file. Exiting.")
        exit(0)

prs_df = prs_df.rename(columns={
    'hm_chr': 'chr',
    'hm_pos': 'bp',
    'effect_allele': 'effect_allele',
    'other_allele': 'noneffect_allele',
    'effect_weight': 'weight'
})[['chr','bp','effect_allele','noneffect_allele','weight']]

prs_df['chr'] = prs_df['chr'].astype(str)
print("First few contigs in PRS weights before any prefix change:")
print(prs_df['chr'].head(10))

# We already have contig_example from VDS
print("VDS contig example:", contig_example)
if contig_example.startswith('chr'):
    print("VDS contig has 'chr' prefix. Ensure PRS also has 'chr' prefix.")
    prs_df['chr'] = prs_df['chr'].apply(lambda c: c if c.startswith('chr') else 'chr'+c)
else:
    print("VDS contig does NOT have 'chr' prefix. Remove 'chr' from PRS if present.")
    prs_df['chr'] = prs_df['chr'].apply(lambda c: c.replace('chr', ''))

print("Contigs in PRS after adjusting for VDS naming:")
print(prs_df['chr'].head(10))

prs_prepared_path = 'PGS003446_prepared_weight_table.csv'
try:
    prs_df.to_csv(prs_prepared_path, index=False)
except Exception as e:
    print(f"Failed to save prepared PRS table: {e}")
    exit(0)

output_prs_weight_path = f'prs_calculation/{prs_identifier}/PGS003446_weight_table.csv'
full_prs_gcs_path = f'{bucket}/{output_prs_weight_path}'
print(f"Uploading PRS weight table to {full_prs_gcs_path}...")
try:
    subprocess.run(["gsutil", "-m", "cp", prs_prepared_path, full_prs_gcs_path], check=True)
except Exception as e:
    print(f"Failed to upload PRS weight table: {e}")
    exit(0)
print("PRS weight table uploaded.")

def import_prs_table():
    print("Importing PRS weight table from CSV in GCS with specified types...")
    if not hl.utils.hadoop_exists(full_prs_gcs_path):
        print(f"PRS file {full_prs_gcs_path} not found in GCS. Exiting.")
        hl.stop()
        exit(0)

    ht = hl.import_table(full_prs_gcs_path, delimiter=',', impute=False,
                         types={'chr': hl.tstr, 'bp': hl.tint32,
                                'effect_allele': hl.tstr, 'noneffect_allele': hl.tstr,
                                'weight': hl.tfloat64})
    print("First few rows of PRS table from hail:")
    ht.show(5)

    ref_genome = 'GRCh38'
    print(f"Using reference genome: {ref_genome}")

    # Check if we have any variants:
    if ht.count() == 0:
        print("No variants in PRS HT after import. Exiting.")
        hl.stop()
        exit(0)

    print("Ensuring contigs match VDS contigs. First few contigs from PRS HT:")
    ht.select(ht.chr).show(5)

    # Attempt creating locus
    # If this fails, we handle it:
    try:
        ht = ht.annotate(locus=hl.locus(ht.chr, ht.bp, reference_genome=ref_genome),
                         alleles=hl.array([ht.noneffect_allele, ht.effect_allele]))
    except Exception as e:
        print(f"Failed to create locus: {e}")
        hl.stop()
        exit(0)

    ht = ht.key_by('locus','alleles')
    return ht

prs_ht_cache = f'{bucket}/prs_calculation/{prs_identifier}/cache/prs_weight_table.ht'
prs_ht = load_or_compute(prs_ht_cache, import_prs_table)

print("Extracting intervals from PRS weight table...")
interval_ht = prs_ht.key_by()

count_prs_vars = interval_ht.count()
if count_prs_vars == 0:
    print("No variants in PRS HT. Exiting.")
    hl.stop()
    exit(0)

some_loci = interval_ht.locus.take(5)
print("Some loci from PRS HT before interval creation:", some_loci)

interval_ht = interval_ht.annotate(
    interval=hl.locus_interval(
        interval_ht.locus.contig,
        interval_ht.locus.position,
        interval_ht.locus.position,
        includes_end=True,
        reference_genome='GRCh38'
    )
)

interval_ht = interval_ht.key_by('interval')
interval_table = interval_ht.select().distinct()

interval_cache = f'{bucket}/prs_calculation/{prs_identifier}/cache/prs_intervals.ht'
interval_table = load_or_compute(interval_cache, lambda: interval_table)

count_intervals = interval_table.count()
print(f"Number of intervals: {count_intervals}")
if count_intervals == 0:
    print("No intervals created. Possibly no variants. Exiting.")
    hl.stop()
    exit(0)

print("Filtering VDS by intervals from PRS weights...")
some_intervals = interval_table.key_by().select('interval').take(5)
print("First few intervals to filter by:", some_intervals)

def filter_by_intervals():
    intervals_list = interval_table.key_by().select('interval').collect()
    print("First few intervals after final step:")
    for i, r in enumerate(intervals_list[:5]):
        print(f"Interval {i}: {r.interval}")

    if len(intervals_list) == 0:
        print("No intervals to filter by. Returning vds_filtered unchanged.")
        return vds_filtered
    return hl.vds.filter_intervals(vds_filtered, [r.interval for r in intervals_list], keep=True)

vds_prs = filter_by_intervals()

# Check if we still have variants after filtering intervals
if vds_prs.variant_data.count_rows() == 0:
    print("No variants remain after interval filtering. Exiting.")
    hl.stop()
    exit(0)

#####################################
# Densify Only This Small Subset
#####################################
print("Densifying only the PRS subset of the VDS...")
def densify_subset():
    try:
        return hl.vds.to_dense_mt(vds_prs)
    except Exception as e:
        print(f"Failed to densify VDS: {e}")
        hl.stop()
        exit(0)

dense_mt_cache = f'{bucket}/prs_calculation/{prs_identifier}/cache/dense_prs.mt'
mt_prs = load_or_compute(dense_mt_cache, densify_subset)

col_count = mt_prs.count_cols()
row_count = mt_prs.count_rows()
print(f"Dense MT samples: {col_count}, variants: {row_count}")
if col_count == 0 or row_count == 0:
    print("No samples or variants after densification. Exiting.")
    hl.stop()
    exit(0)

#####################################
# Annotate with PRS and Compute
#####################################
print("Annotating MT with PRS weights...")
if prs_ht.count() == 0:
    print("PRS HT empty. Exiting.")
    hl.stop()
    exit(0)

mt_prs = mt_prs.annotate_rows(prs_weight = prs_ht[mt_prs.row_key].weight)
print("Computing dosage and PRS contribution...")

mt_prs = mt_prs.annotate_entries(
    dosage = hl.cond(hl.is_defined(mt_prs.GT), mt_prs.GT.n_alt_alleles(), 0),
    prs_contrib = hl.cond(hl.is_defined(mt_prs.prs_weight), mt_prs.dosage * mt_prs.prs_weight, 0.0)
)

print("Aggregating PRS per sample...")
mt_prs = mt_prs.annotate_cols(
    sum_weights=hl.agg.sum(mt_prs.prs_contrib),
    N_variants=hl.agg.count_where(hl.is_defined(mt_prs.prs_weight))
)

scores_output_path = f'{bucket}/prs_calculation/{prs_identifier}/scores/PGS003446_scores.csv'
if not hl.utils.hadoop_exists(scores_output_path):
    print(f"Exporting PRS scores to {scores_output_path}...")
    try:
        mt_prs.cols().export(scores_output_path)
    except Exception as e:
        print(f"Failed to export PRS scores: {e}")
        # We do not exit because we can still analyze
else:
    print("Scores already exported previously, skipping re-export.")

print("PRS scores file at:", scores_output_path)
if not hl.utils.hadoop_exists(scores_output_path):
    print("Scores CSV not found. Possibly failed to export. Exiting.")
    hl.stop()
    exit(0)

#####################################
# Post-processing
#####################################
print("Reading PRS scores file for analysis...")
try:
    scores_df = pd.read_csv(scores_output_path)
except Exception as e:
    print(f"Failed to read scores file: {e}")
    hl.stop()
    exit(0)

if scores_df.empty:
    print("Scores dataframe is empty. Exiting.")
    hl.stop()
    exit(0)

print("Head of scores_df:")
print(scores_df.head())
print("Score statistics:\n", scores_df['sum_weights'].describe())
print("Number of variants contributing per sample:\n", scores_df['N_variants'].value_counts())

if scores_df['sum_weights'].std() == 0:
    print("All sum_weights identical, cannot compute z-scores. Exiting gracefully.")
    hl.stop()
    exit(0)

print("Standardizing PRS scores...")
scores_df['prs_zscore'] = (scores_df['sum_weights'] - scores_df['sum_weights'].mean()) / scores_df['sum_weights'].std()
scores_df['prs_percentile'] = scores_df['prs_zscore'].rank(pct=True)*100
try:
    scores_df['risk_category'] = pd.qcut(scores_df['prs_zscore'], q=5, labels=['Very Low','Low','Average','High','Very High'])
except ValueError:
    print("Not enough distinct values to create 5 quintiles. Assigning single category.")
    scores_df['risk_category'] = 'Average'

standardized_scores_path = f'{bucket}/prs_calculation/{prs_identifier}/scores/PGS003446_scores_standardized.csv'
if not hl.utils.hadoop_exists(standardized_scores_path):
    print(f"Saving standardized scores to {standardized_scores_path}...")
    try:
        scores_df.to_csv(standardized_scores_path, index=False)
    except Exception as e:
        print(f"Failed to save standardized scores: {e}")
else:
    print(f"Standardized scores file already exists at {standardized_scores_path}, not overwriting.")

print("Plotting PRS score distributions...")
if 'prs_zscore' in scores_df.columns and not scores_df['prs_zscore'].empty:
    plt.figure(figsize=(15,10))

    plt.subplot(2, 1, 1)
    sns.histplot(data=scores_df, x='prs_zscore', bins=50)
    plt.title('Distribution of PRS Z-scores')
    plt.xlabel('PRS Z-score')
    plt.ylabel('Count')

    plt.subplot(2, 1, 2)
    # Check if multiple categories exist:
    if scores_df['risk_category'].nunique() > 1:
        sns.boxplot(data=scores_df, x='risk_category', y='prs_zscore')
        plt.title('PRS Z-scores by Risk Category')
        plt.xticks(rotation=45)
    else:
        print("Only one risk category, skipping boxplot.")
    plt.tight_layout()

    try:
        plt.savefig("prs_distribution_plots.png")
        print("PRS distribution plots saved as prs_distribution_plots.png")
    except Exception as e:
        print(f"Failed to save plots: {e}")
else:
    print("No z-scores available for plotting, skipping.")

print("Script completed successfully.")
end_time = datetime.datetime.now()
print("End date:", end_time.strftime("%Y-%m-%d"))
print("End time:", end_time.strftime("%H:%M:%S"))

# Graceful stop
hl.stop()

exit(0)
