import datetime
import os
import hail as hl
import pandas as pd
import gcsfs
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import scipy.stats as stats
import numpy as np

# Record start time
job_initiation_time = datetime.datetime.now()
print("********** JOB STARTING **********")
print("Workflow start date:", job_initiation_time.strftime("%Y-%m-%d"))
print("Workflow start time:", job_initiation_time.strftime("%H:%M:%S"))
print("Commencing the PRS analysis workflow...")
print("***********************************\n")

print("All libraries imported successfully.")

# Caching only for DataFrame results
CACHE_DIR = "local_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(name):
    return f"{CACHE_DIR}/{name}.pkl"

def load_from_cache(name):
    path = cache_path(name)
    if os.path.exists(path):
        return pd.read_pickle(path)
    return None

def cache_result(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cached = load_from_cache(name)
            if cached is not None:
                print(f"[CACHE HIT] Loading '{name}' from cache.")
                return cached
            print(f"[CACHE MISS] Running step '{name}'...")
            result = func(*args, **kwargs)
            pd.to_pickle(result, cache_path(name))
            return result
        return wrapper
    return decorator

project_bucket = os.getenv("WORKSPACE_BUCKET")
print("Detected workspace bucket:", project_bucket, "\n")

print("Configuring Hail environment with specified temp directory and reference genome...")
hl.init(tmp_dir=f'{project_bucket}/hail_temp/', default_reference='GRCh38')
print("Hail initialized successfully.\n")

print("Defining primary input paths...")
wgs_vds_path = 'gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/vds/hail.vds'
flagged_samples_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"
print("WGS VDS path:", wgs_vds_path)
print("Flagged samples table path:", flagged_samples_path, "\n")





def load_full_vds():
    print("Loading full WGS VariantDataset...")
    vds = hl.vds.read_vds(wgs_vds_path)
    print("VDS successfully loaded.")
    print("Initial VDS sample count:", vds.n_samples(), "\n")
    return vds

def load_excluded_samples_ht():
    print("Importing flagged (related or excluded) samples...")
    ht = hl.import_table(flagged_samples_path, key='sample_id')
    print("Flagged samples loaded.\n")
    return ht

def filter_samples(vds, excluded_ht):
    print("Filtering flagged samples out of the VDS...")
    cleaned = hl.vds.filter_samples(vds, excluded_ht, keep=False)
    print("Samples filtered. Post-filter sample count:", cleaned.n_samples(), "\n")
    return cleaned

@cache_result("wgs_ehr_samples_df")
def get_wgs_ehr_samples_df():
    print("Retrieving WGS+EHR samples from BigQuery...")
    workspace_cdr = os.environ["WORKSPACE_CDR"]
    wgs_ehr_query = f"""
    SELECT person_id
    FROM `{workspace_cdr}.person`
    WHERE person_id IN (
        SELECT DISTINCT person_id
        FROM `{workspace_cdr}.cb_search_person`
        WHERE has_ehr_data = 1
        AND has_whole_genome_variant = 1
    )
    """
    df = pd.read_gbq(wgs_ehr_query, dialect='standard')
    print("WGS+EHR query completed. Number with WGS+EHR:", df['person_id'].nunique(), "\n")
    return df

def save_wgs_ehr_ids(df):
    wgs_ehr_ids_csv = f'{project_bucket}/prs_calculator_tutorial/people_with_WGS_EHR_ids.csv'
    print("Storing WGS+EHR sample IDs to:", wgs_ehr_ids_csv)
    df.to_csv(wgs_ehr_ids_csv, index=False)
    print("WGS+EHR sample IDs saved.\n")
    return wgs_ehr_ids_csv

def filter_vds_to_wgs_ehr(vds, wgs_ehr_ids_csv):
    print("Importing WGS+EHR sample list for VDS filtering...")
    wgs_ehr_ht = hl.import_table(wgs_ehr_ids_csv, delimiter=',', key='person_id')
    subset = hl.vds.filter_samples(vds, wgs_ehr_ht, keep=True)
    print("VDS filtered to WGS+EHR samples. Final count:", subset.n_samples(), "\n")
    return subset

def download_prs_files(prs_id, harmonized_url):
    # Removed the original_url since it was unused
    harmonized_local = f'{prs_id}_hmPOS_GRCh38.txt.gz'

    print("Downloading harmonized PRS file from:", harmonized_url)
    response = requests.get(harmonized_url)
    with open(harmonized_local, 'wb') as out_file:
        out_file.write(response.content)
    print("Harmonized PRS weight file downloaded.\n")

    print("Listing local files for verification...")
    !ls
    print("\nDecompressing harmonized PRS file for inspection...")
    !gunzip -f {harmonized_local}
    return harmonized_local.replace('.gz', '')

def preview_harmonized_file(harmonized_file):
    print("Previewing the first 25 lines of the harmonized PRS file:")
    !head -n 25 {harmonized_file}
    print("Preview complete.\n")

def prepare_prs_weight_table(harmonized_file, prs_id):
    print("Reading harmonized PRS weight file into DataFrame...")
    score_df = pd.read_csv(harmonized_file, sep='\t', comment='#')
    print("PRS weight data loaded. Variant count:", len(score_df), "\n")

    print("Renaming and selecting relevant columns...")
    score_df = score_df.rename(columns={
        'hm_chr': 'chr',
        'hm_pos': 'bp',
        'effect_allele': 'effect_allele',
        'effect_weight': 'weight'
    })

    # Necessary columns
    score_df = score_df[['chr', 'bp', 'effect_allele', 'weight']]
    score_df['chr'] = score_df['chr'].astype(str)

    prepared_csv = f'{prs_id}_prepared_weight_table.csv'
    print("Storing processed PRS weight table locally as:", prepared_csv)
    score_df.to_csv(prepared_csv, index=False)
    print("PRS weight table saved locally.\n")
    return prepared_csv

def upload_prs_table_to_gcs(local_csv, prs_id):
    prs_weights_dest = f'scores/{prs_id}/weight_table/{prs_id}_weight_table.csv'
    print("Copying prepared PRS weight table to GCS:", f"{project_bucket}/{prs_weights_dest}")
    !gsutil cp {local_csv} {project_bucket}/{prs_weights_dest}
    print("Copy to GCS completed.\n")
    return prs_weights_dest

@cache_result("rebuilt_prs_table_df")
def rebuild_prs_weight_table_as_df(input_path, bucket):
    print("\n[ Rebuilding PRS Weight Table for Analysis ]")
    if bucket:
        fs = gcsfs.GCSFileSystem()
        with fs.open(f'{bucket}/{input_path}', 'rb') as infile:
            df = pd.read_csv(infile)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded PRS weights: {df.shape[0]} variants, {df.shape[1]} columns.")
    required_cols = ['chr', 'bp']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Adding Hail-compatible columns: contig, position")
    df['chr'] = df['chr'].astype(str)
    df['contig'] = df['chr'].apply(lambda c: c if c.startswith('chr') else f'chr{c}')
    df['position'] = df['bp']
    print("[ PRS Weight Table Rebuild Complete ]\n")
    return df

def save_rebuilt_score_df_to_gcs(df, output_path, bucket):
    print("Saving updated PRS weight table...")
    fs = gcsfs.GCSFileSystem() if bucket else None
    if bucket:
        with fs.open(f'{bucket}/{output_path}', 'w') as outfile:
            df.to_csv(outfile, index=False)
        print("PRS weight table saved to GCS.")
    else:
        df.to_csv(output_path, index=False)
        print("PRS weight table saved locally.")

def load_prepared_prs_table_for_annotation(bucket, weight_path):
    print("Re-importing PRS weights for annotation...")
    fs = gcsfs.GCSFileSystem() if bucket else None
    if bucket:
        with fs.open(f'{bucket}/{weight_path}', 'rb') as infile:
            score_df = pd.read_csv(infile)
    else:
        score_df = pd.read_csv(weight_path)

    essential = ["weight", "contig", "position", "effect_allele"]
    for c in essential:
        if c not in score_df.columns:
            raise ValueError(f"PRS weight table missing required column: {c}")

    fs = gcsfs.GCSFileSystem()
    temp_gcs = f"{project_bucket}/hail_temp/temp_prs_for_hail.csv"
    with fs.open(temp_gcs, 'w') as outfile:
        score_df.to_csv(outfile, index=False)

    prs_ht = hl.import_table(temp_gcs, delimiter=',', types={'weight': hl.tfloat64, 'position': hl.tint32})
    prs_ht = prs_ht.annotate(locus=hl.locus(prs_ht.contig, prs_ht.position))
    prs_ht = prs_ht.key_by('locus')
    print("PRS HT ready for annotation.\n")
    return prs_ht

def calculate_effect_allele_dosage(vds_row):
    eff_allele = vds_row.prs_variant_info['effect_allele']
    ref_allele = vds_row.alleles[0]
    alt_alleles = hl.set(vds_row.alleles[1:].map(lambda x: x))

    is_effect_ref = (ref_allele == eff_allele)
    is_effect_alt = alt_alleles.contains(eff_allele)

    return (hl.case()
             .when(hl.is_missing(vds_row.GT) & is_effect_ref, 2)
             .when(hl.is_missing(vds_row.GT) & is_effect_alt, 0)
             .when(vds_row.GT.is_hom_ref() & is_effect_ref, 2)
             .when(vds_row.GT.is_hom_var() & is_effect_alt, 2)
             .when(vds_row.GT.is_het() & is_effect_ref, 1)
             .when(vds_row.GT.is_het() & is_effect_alt, 1)
             .default(0))

def extract_intervals_for_filter(score_df, bucket, output_path, prs_id):
    print("Extracting variant intervals for filtering VDS...")
    score_df['end'] = score_df['position']
    intervals_df = score_df[['contig', 'position', 'end']]

    interval_file = f"{output_path}/interval/{prs_id}_interval.tsv"
    full_interval_path = f"{bucket}/{interval_file}" if bucket else interval_file
    print("Saving interval list to:", full_interval_path)
    fs = gcsfs.GCSFileSystem() if bucket else None
    if bucket:
        with fs.open(full_interval_path, 'w') as outfile:
            intervals_df.to_csv(outfile, header=False, index=False, sep="\t")
    else:
        intervals_df.to_csv(full_interval_path, header=False, index=False, sep="\t")
    print("Intervals saved successfully.\n")
    return full_interval_path

def filter_vds_by_intervals(vds, interval_path):
    print("Filtering VDS by PRS intervals...")
    locus_intervals = hl.import_locus_intervals(interval_path, reference_genome='GRCh38', skip_invalid_intervals=True)
    vds_filtered = hl.vds.filter_intervals(vds, locus_intervals, keep=True)
    print("VDS filtered to PRS intervals.\n")
    return vds_filtered

def annotate_and_compute_prs_scores(vds_filtered, prs_ht):
    print("Annotating VDS with PRS info and computing scores...")
    prs_mt = vds_filtered.variant_data.annotate_rows(prs_variant_info=prs_ht[vds_filtered.variant_data.locus])
    prs_mt = prs_mt.unfilter_entries()

    effect_allele_dosage_expr = calculate_effect_allele_dosage(prs_mt)
    prs_mt = prs_mt.annotate_entries(
        effect_allele_count=effect_allele_dosage_expr,
        variant_contribution=effect_allele_dosage_expr * prs_mt.prs_variant_info['weight']
    )
    prs_mt = prs_mt.annotate_cols(
        total_score=hl.agg.sum(prs_mt.variant_contribution),
        variant_count=hl.agg.count_where(hl.is_defined(prs_mt.variant_contribution))
    )

    # Normalize by the number of variants before z-scoring
    prs_mt = prs_mt.annotate_cols(
        normalized_score=prs_mt.total_score / hl.float64(prs_mt.variant_count)
    )

    print("PRS scores computed and normalized.\n")
    return prs_mt

def save_prs_scores(prs_mt, output_path, prs_id, bucket):
    hail_dir = f'{output_path}/hail/'
    score_csv = f'{output_path}/score/{prs_id}_scores.csv'
    full_hail_dir = f'{bucket}/{hail_dir}' if bucket else hail_dir
    full_score_csv = f'{bucket}/{score_csv}' if bucket else score_csv

    print("Writing PRS results to Hail Table:", full_hail_dir)
    prs_mt.key_cols_by().cols().write(full_hail_dir, overwrite=True)
    print("Hail Table write complete.\n")

    print("Exporting PRS scores to CSV:", full_score_csv)
    final_ht = hl.read_table(full_hail_dir)
    final_ht.export(full_score_csv, header=True, delimiter=',')
    print("PRS scores exported successfully.\n")
    return full_score_csv

def export_found_variants(prs_mt, output_path, prs_id, bucket):
    found_variants_csv = f'{output_path}/score/{prs_id}_found_in_aou.csv'
    full_found_csv = f'{bucket}/{found_variants_csv}' if bucket else found_variants_csv
    print("Extracting variants found in AoU cohort...")
    found_vars_ht = prs_mt.filter_rows(hl.is_defined(prs_mt.prs_variant_info)).rows()
    found_vars_df = found_vars_ht.select(found_vars_ht.prs_variant_info).to_pandas()
    print("Number of variants found in AoU:", found_vars_df.shape[0])

    fs = gcsfs.GCSFileSystem() if bucket else None
    if bucket:
        with fs.open(full_found_csv, 'w') as outfile:
            found_vars_df.to_csv(outfile, header=True, index=False, sep=',')
    else:
        found_vars_df.to_csv(full_found_csv, header=True, index=False, sep=',')
    print("Found variants at:", full_found_csv, "\n")


# ------------------- MAIN SCRIPT EXECUTION -------------------
full_vds = load_full_vds()
excluded_ht = load_excluded_samples_ht()
cleaned_vds = filter_samples(full_vds, excluded_ht)

wgs_ehr_df = get_wgs_ehr_samples_df()
wgs_ehr_ids_csv = save_wgs_ehr_ids(wgs_ehr_df)
prs_subset_vds = filter_vds_to_wgs_ehr(cleaned_vds, wgs_ehr_ids_csv)

target_prs_id = 'PGS004162'
harmonized_url = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS004162/ScoringFiles/Harmonized/PGS004162_hmPOS_GRCh38.txt.gz'
harmonized_file = download_prs_files(target_prs_id, harmonized_url)
preview_harmonized_file(harmonized_file)
prepared_csv = prepare_prs_weight_table(harmonized_file, target_prs_id)
prs_weights_destination = upload_prs_table_to_gcs(prepared_csv, target_prs_id)

rebuilt_df = rebuild_prs_weight_table_as_df(prs_weights_destination, project_bucket)
save_rebuilt_score_df_to_gcs(rebuilt_df, prs_weights_destination, project_bucket)
print("PRS weight table preparation phase finished.\n")

prs_output_directory = f'scores/{target_prs_id}/calculated_scores'
intervals_path = extract_intervals_for_filter(rebuilt_df, project_bucket, prs_output_directory, target_prs_id)
vds_filtered = filter_vds_by_intervals(prs_subset_vds, intervals_path)
prs_ht = load_prepared_prs_table_for_annotation(project_bucket, prs_weights_destination)
prs_mt = annotate_and_compute_prs_scores(vds_filtered, prs_ht)
score_csv_path = save_prs_scores(prs_mt, prs_output_directory, target_prs_id, project_bucket)
export_found_variants(prs_mt, prs_output_directory, target_prs_id, project_bucket)

print("PRS computation workflow finished.\n")

calculation_end = datetime.datetime.now()
print("PRS calculation step finished at date:", calculation_end.strftime("%Y-%m-%d"))
print("PRS calculation step finished at time:", calculation_end.strftime("%H:%M:%S"), "\n")

prs_scores_gcs = f'{project_bucket}/scores/{target_prs_id}/calculated_scores/score/{target_prs_id}_scores.csv'
found_variants_gcs = f'{project_bucket}/scores/{target_prs_id}/calculated_scores/score/{target_prs_id}_found_in_aou.csv'

print("Loading final PRS scores from:", prs_scores_gcs)
prs_scores_final_df = pd.read_csv(prs_scores_gcs)
print("Loaded PRS scores. Shape:", prs_scores_final_df.shape)

print("Loading variants discovered in AoU dataset from:", found_variants_gcs)
found_variants_final_df = pd.read_csv(found_variants_gcs)
print("Loaded discovered variants. Shape:", found_variants_final_df.shape, "\n")

print("Descriptive stats for PRS scores:")
print(prs_scores_final_df['total_score'].describe(), "\n")

print("Distribution of the count of variants per sample:")
print(prs_scores_final_df['variant_count'].value_counts(), "\n")

print("Computing Z-scores and percentiles for the PRS distribution...")
prs_scores_final_df['prs_zscore'] = (prs_scores_final_df['total_score'] - prs_scores_final_df['total_score'].mean()) / prs_scores_final_df['total_score'].std()
prs_scores_final_df['prs_percentile'] = prs_scores_final_df['prs_zscore'].rank(pct=True) * 100

print("Categorizing samples into quintiles...")
prs_scores_final_df['risk_category'] = pd.qcut(prs_scores_final_df['prs_zscore'], q=5, labels=['Very Low', 'Low', 'Average', 'High', 'Very High'])

sns.set_style("whitegrid")
sns.set_context("talk")
palette = sns.color_palette("viridis", as_cmap=True)

fig, axs = plt.subplots(2, 2, figsize=(20, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Histogram of PRS z-scores
sns.histplot(data=prs_scores_final_df, x='prs_zscore', bins=50, ax=axs[0,0], color=palette(0.2))
axs[0,0].set_title('T1D PRS Z-score Distribution', fontsize=18, fontweight='bold')
axs[0,0].set_xlabel('PRS Z-score', fontsize=16)
axs[0,0].set_ylabel('Frequency', fontsize=16)

# Boxplot by risk category
sns.boxplot(data=prs_scores_final_df, x='risk_category', y='prs_zscore', ax=axs[0,1], palette="magma")
axs[0,1].set_title('PRS Z-score by Risk Category', fontsize=18, fontweight='bold')
axs[0,1].set_xlabel('Risk Category', fontsize=16)
axs[0,1].set_ylabel('PRS Z-score', fontsize=16)
axs[0,1].tick_params(axis='x', rotation=45)

# KDE plot of PRS z-scores
sns.kdeplot(data=prs_scores_final_df, x='prs_zscore', shade=True, ax=axs[1,0], color=palette(0.6))
axs[1,0].set_title('PRS Z-score Density', fontsize=18, fontweight='bold')
axs[1,0].set_xlabel('PRS Z-score', fontsize=16)
axs[1,0].set_ylabel('Density', fontsize=16)

# Scatterplot: Z-score vs Percentile (just to check)
sns.scatterplot(data=prs_scores_final_df, x='prs_zscore', y='prs_percentile', ax=axs[1,1], color=palette(0.8), alpha=0.7)
axs[1,1].set_title('PRS Z-score vs Percentile', fontsize=18, fontweight='bold')
axs[1,1].set_xlabel('PRS Z-score', fontsize=16)
axs[1,1].set_ylabel('PRS Percentile', fontsize=16)

plt.suptitle("T1D PRS Distribution and Categories", fontsize=24, fontweight='bold')
plt.show()

print("Z-score stats:")
print(prs_scores_final_df['prs_zscore'].describe(), "\n")
print("Counts in each PRS risk category:")
print(prs_scores_final_df['risk_category'].value_counts().sort_index(), "\n")

standardized_scores_path = f'{project_bucket}/scores/{target_prs_id}/calculated_scores/score/{target_prs_id}_scores_standardized.csv'
print("Saving standardized PRS scores (with z-scores and categories) to:", standardized_scores_path)
prs_scores_final_df.to_csv(standardized_scores_path, index=False)
print("Standardized PRS scores saved.\n")

print("Evaluating PRS against T1D phenotype...")
workspace_cdr = os.environ["WORKSPACE_CDR"]

T1D_query = f"""
SELECT 
    person_id AS s,
    MAX(
        CASE WHEN condition_source_value IN (
            -- ICD-10 codes for T1D
            'E10', 'E10.0', 'E10.1', 'E10.2', 'E10.3', 'E10.4', 'E10.5', 
            'E10.6', 'E10.7', 'E10.8', 'E10.9',
            -- ICD-9 codes for T1D
            '250.01', '250.03', '250.11', '250.13', '250.21', '250.23', 
            '250.31', '250.33', '250.41', '250.43', '250.51', '250.53', 
            '250.61', '250.63', '250.71', '250.73', '250.81', '250.83', 
            '250.91', '250.93'
        )
        THEN 1 
        ELSE 0 
        END
    ) AS T1D_status
FROM `{workspace_cdr}.condition_occurrence`
GROUP BY person_id
"""
T1D_phenotype = pd.read_gbq(T1D_query, dialect='standard')
print("T1D phenotype data retrieved. Shape:", T1D_phenotype.shape, "\n")



print("Merging PRS scores with T1D phenotype data...")
merged_scores = prs_scores_final_df.merge(T1D_phenotype, on='s', how='inner')
print("Merged DataFrame shape:", merged_scores.shape, "\n")

print("Calculating ROC-AUC...")
auc_val = roc_auc_score(merged_scores['T1D_status'], merged_scores['prs_zscore'])
fpr_arr, tpr_arr, _ = roc_curve(merged_scores['T1D_status'], merged_scores['prs_zscore'])

plt.figure(figsize=(10, 8))
sns.lineplot(x=fpr_arr, y=tpr_arr, color='blue', lw=2, label=f'ROC (AUC = {auc_val:.3f})')
sns.lineplot(x=[0,1], y=[0,1], color='gray', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('T1D PRS ROC Curve', fontsize=18, fontweight='bold')
plt.legend(loc="lower right")
plt.show()



print("Analyzing T1D prevalence across PRS deciles...")

# Categorize samples into deciles based on PRS z-score
merged_scores['prs_decile'] = pd.qcut(merged_scores['prs_zscore'], q=10, labels=[f'D{i}' for i in range(1,11)])

# Compute prevalence of T1D in each decile
decile_stats = merged_scores.groupby('prs_decile').agg({'T1D_status': ['count', 'mean']}).reset_index()
decile_stats.columns = ['prs_decile', 'count', 'prevalence']
decile_stats['prevalence'] = decile_stats['prevalence'] * 100

print("T1D Prevalence by PRS Decile:")
print(decile_stats, "\n")

# Plot T1D prevalence by decile
plt.figure(figsize=(12, 6))
sns.barplot(data=decile_stats, x='prs_decile', y='prevalence', palette='viridis')
plt.title('T1D Prevalence by PRS Decile', fontsize=18, fontweight='bold')
plt.xlabel('PRS Decile', fontsize=14)
plt.ylabel('T1D Prevalence (%)', fontsize=14)
for i, row in decile_stats.iterrows():
    plt.text(i, row['prevalence']+0.5, f"{row['prevalence']:.2f}%", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()

# ---------------------------------------
# CHANGE IN RISK PER SD INCREMENT
# ---------------------------------------

print("Calculating odds ratio per standard deviation (SD) increase in PRS...")

# Logistic regression with PRS z-score as a predictor of T1D status
X = merged_scores[['prs_zscore']]
y = merged_scores['T1D_status']
log_model = LogisticRegression()
log_model.fit(X, y)

# Extract odds ratio per SD from the logistic model coefficients
# The logistic regression coefficient is in log-odds per unit change in PRS z-score.
# Since PRS z-score is already standardized, the coefficient directly represents log-odds per SD.
log_odds_per_SD = log_model.coef_[0][0]
odds_ratio_per_SD = np.exp(log_odds_per_SD)

print(f"Odds Ratio per 1 SD increase in PRS: {odds_ratio_per_SD:.2f}\n")



merged_scores['prs_quintile'] = pd.qcut(merged_scores['prs_zscore'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
quintile_stats = merged_scores.groupby('prs_quintile').agg({'T1D_status': ['count', 'mean']}).reset_index()

print("=== T1D Association Summary ===")
print("ROC-AUC:", round(auc_val, 3))
print("\nT1D prevalence by PRS quintile:")
print(quintile_stats, "\n")

print("Performing additional T1D prevalence and logistic regression analysis...")
total_subjects = len(merged_scores)
total_T1D_cases = merged_scores['T1D_status'].sum()
T1D_overall_prevalence = merged_scores['T1D_status'].mean()*100

print("Overall T1D prevalence in analyzed cohort:")
print(f"Total participants: {total_subjects}")
print(f"T1D cases: {total_T1D_cases}")
print(f"T1D prevalence: {T1D_overall_prevalence:.2f}%\n")

X = merged_scores['prs_zscore'].values.reshape(-1, 1)
y = merged_scores['T1D_status'].values
log_model = LogisticRegression()
log_model.fit(X, y)
p_value_approx = stats.chi2.sf(log_model.score(X, y)*len(y), df=1)
print("Logistic Regression test p-value:", f"{p_value_approx:.2e}\n")

top_5_threshold = np.percentile(merged_scores['prs_zscore'], 95)
bottom_5_threshold = np.percentile(merged_scores['prs_zscore'], 5)

lowest_5 = merged_scores[merged_scores['prs_zscore'] <= bottom_5_threshold]
highest_5 = merged_scores[merged_scores['prs_zscore'] >= top_5_threshold]

comparison_data = {
    'Bottom 5%': {
        'prevalence': lowest_5['T1D_status'].mean()*100,
        'count': len(lowest_5),
        'cases': lowest_5['T1D_status'].sum()
    },
    'Top 5%': {
        'prevalence': highest_5['T1D_status'].mean()*100,
        'count': len(highest_5),
        'cases': highest_5['T1D_status'].sum()
    }
}

plt.figure(figsize=(10, 6))
group_labels = ['Bottom 5%', 'Top 5%']
group_prevalences = [comparison_data[g]['prevalence'] for g in group_labels]
bars = plt.bar(group_labels, group_prevalences, color=['lightblue', 'darkblue'])

for idx, bar_obj in enumerate(bars):
    height = bar_obj.get_height()
    grp = group_labels[idx]
    plt.text(bar_obj.get_x() + bar_obj.get_width()/2., height,
             f'Cases={comparison_data[grp]["cases"]}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('T1D Prevalence in Top vs Bottom 5% of PRS Distribution', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('T1D Prevalence (%)', fontsize=14)
plt.tight_layout()
plt.show()

top_5_odds = comparison_data['Top 5%']['cases'] / (comparison_data['Top 5%']['count'] - comparison_data['Top 5%']['cases'])
bottom_5_odds = comparison_data['Bottom 5%']['cases'] / (comparison_data['Bottom 5%']['count'] - comparison_data['Bottom 5%']['cases'])
odds_ratio_val = top_5_odds / bottom_5_odds

print("=== High vs Low PRS Comparison ===")
for grp_key, grp_val in comparison_data.items():
    print(f"\n{grp_key}:")
    print(f"  Prevalence: {grp_val['prevalence']:.2f}%")
    print(f"  Total N: {grp_val['count']}")
    print(f"  T1D cases: {grp_val['cases']}")

print(f"\nOdds Ratio (Top 5% vs Bottom 5%): {odds_ratio_val:.2f}\n")

job_end_time = datetime.datetime.now()
print("********** JOB COMPLETED **********")
print("Completion date:", job_end_time.strftime("%Y-%m-%d"))
print("Completion time:", job_end_time.strftime("%H:%M:%S"))
print("All steps finished successfully.")
print("***********************************\n")


import matplotlib.pyplot as plt
import seaborn as sns

# Set up aesthetics
sns.set_style("whitegrid")
sns.set_context("talk")
palette = sns.color_palette("viridis", as_cmap=True)

# Recreate PRS distribution plots
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# Histogram of PRS z-scores
sns.histplot(data=prs_scores_final_df, x='prs_zscore', bins=50, ax=axs[0,0], color=palette(0.2))
axs[0,0].set_title('T1D PRS Z-score Distribution', fontsize=18, fontweight='bold')
axs[0,0].set_xlabel('PRS Z-score', fontsize=16)
axs[0,0].set_ylabel('Frequency', fontsize=16)

# Boxplot by risk category
sns.boxplot(data=prs_scores_final_df, x='risk_category', y='prs_zscore', ax=axs[0,1], palette="magma")
axs[0,1].set_title('PRS Z-score by Risk Category', fontsize=18, fontweight='bold')
axs[0,1].set_xlabel('Risk Category', fontsize=16)
axs[0,1].set_ylabel('PRS Z-score', fontsize=16)
axs[0,1].tick_params(axis='x', rotation=45)

# KDE plot of PRS z-scores
sns.kdeplot(data=prs_scores_final_df, x='prs_zscore', shade=True, ax=axs[1,0], color=palette(0.6))
axs[1,0].set_title('PRS Z-score Density', fontsize=18, fontweight='bold')
axs[1,0].set_xlabel('PRS Z-score', fontsize=16)
axs[1,0].set_ylabel('Density', fontsize=16)

# Scatterplot: Z-score vs Percentile
sns.scatterplot(data=prs_scores_final_df, x='prs_zscore', y='prs_percentile', ax=axs[1,1], color=palette(0.8), alpha=0.7)
axs[1,1].set_title('PRS Z-score vs Percentile', fontsize=18, fontweight='bold')
axs[1,1].set_xlabel('PRS Z-score', fontsize=16)
axs[1,1].set_ylabel('PRS Percentile', fontsize=16)

plt.suptitle("T1D PRS Distribution and Categories", fontsize=24, fontweight='bold')
plt.show()

# Print summary statistics again
print("Z-score stats:")
print(prs_scores_final_df['prs_zscore'].describe(), "\n")

print("Counts in each PRS risk category:")
print(prs_scores_final_df['risk_category'].value_counts().sort_index(), "\n")

# ROC curve plot
plt.figure(figsize=(10, 8))
sns.lineplot(x=fpr_arr, y=tpr_arr, color='blue', lw=2, label=f'ROC (AUC = {auc_val:.3f})')
sns.lineplot(x=[0,1], y=[0,1], color='gray', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('T1D PRS ROC Curve', fontsize=18, fontweight='bold')
plt.legend(loc="lower right")
plt.show()

print("ROC-AUC:", round(auc_val, 3), "\n")

# T1D prevalence by PRS decile
print("T1D Prevalence by PRS Decile:")
print(decile_stats, "\n")

plt.figure(figsize=(12, 6))
sns.barplot(data=decile_stats, x='prs_decile', y='prevalence', palette='viridis')
plt.title('T1D Prevalence by PRS Decile', fontsize=18, fontweight='bold')
plt.xlabel('PRS Decile', fontsize=14)
plt.ylabel('T1D Prevalence (%)', fontsize=14)
for i, row in decile_stats.iterrows():
    plt.text(i, row['prevalence']+0.5, f"{row['prevalence']:.2f}%", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()

print("=== High vs Low PRS Comparison ===")
for grp_key, grp_val in comparison_data.items():
    print(f"\n{grp_key}:")
    print(f"  Prevalence: {grp_val['prevalence']:.2f}%")
    print(f"  Total N: {grp_val['count']}")
    print(f"  T1D cases: {grp_val['cases']}")

print(f"\nOdds Ratio (Top 5% vs Bottom 5%): {odds_ratio_val:.2f}\n")

# Plot comparison of top vs bottom 5%
plt.figure(figsize=(10, 6))
group_labels = ['Bottom 5%', 'Top 5%']
group_prevalences = [comparison_data[g]['prevalence'] for g in group_labels]
bars = plt.bar(group_labels, group_prevalences, color=['lightblue', 'darkblue'])

for idx, bar_obj in enumerate(bars):
    height = bar_obj.get_height()
    grp = group_labels[idx]
    plt.text(bar_obj.get_x() + bar_obj.get_width()/2., height,
             f'Cases={comparison_data[grp]["cases"]}',
             ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('T1D Prevalence in Top vs Bottom 5% of PRS Distribution', fontsize=18, fontweight='bold', pad=20)
plt.ylabel('T1D Prevalence (%)', fontsize=14)
plt.tight_layout()
plt.show()



# Import sample IDs into Hail Table
sample_needed_ht = hl.import_table(sample_ids_path, delimiter=',', key='person_id')
# Filter samples
vds_subset = hl.vds.filter_samples(cleaned_vds, sample_needed_ht, keep=True)

our_interval = hl.parse_locus_interval("chr6:31283525-31306849")
vds_subset = hl.vds.filter_intervals(vds_subset, [our_interval])
print("Number of samples after filtering for WGS and EHR data:", vds_subset.n_samples())

# vds = vds_subset
# chr6_vds = hl.vds.filter_intervals(vds, [hl.parse_locus_interval('chr6')])

n_rows = vds_subset.variant_data.count_rows()
print(f"Number of rows: {n_rows}")

vd = vds_subset.variant_data
import os

sample_ids = vd.s.collect()  # Collects all sample IDs

# Output directory for individual VCFs
output_dir = f'{project_bucket}/sample_vcfs'
os.makedirs(output_dir, exist_ok=True)

# Loop through each sample and write a distinct VCF
for sample_id in sample_ids:
    # Filter MatrixTable to this sample
    sample_mt = vd.filter_cols(vd.s == sample_id)
    sample_mt = sample_mt.annotate_entries(FT=hl.if_else(sample_mt.FT, 'PASS', 'FAIL'))
    
    # Write the sample's data to a .vcf.gz file
    output_path = os.path.join(output_dir, f'{sample_id}.vcf.bgz')
    hl.export_vcf(sample_mt, output_path)


import json
from collections import defaultdict
import multiprocessing as mp
from itertools import islice
import os

def process_chunk(chunk):
    nodes = defaultdict(set)
    for record in chunk:
        if path := record.get('path'):
            if mapping := path.get('mapping'):
                name = record.get('name')
                if name:
                    nodes[name].update(
                        m['position']['node_id']
                        for m in mapping
                        if 'position' in m and 'node_id' in m['position']
                    )
    return dict(nodes)

def merge_results(results):
    merged = defaultdict(set)
    for result in results:
        for name, nodes in result.items():
            merged[name].update(nodes)
    return merged

def read_json_in_chunks(file, chunk_size=1000):
    chunk = []
    for line in file:
        try:
            record = json.loads(line)
            chunk.append(record)
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        except json.JSONDecodeError:
            continue
    if chunk:
        yield chunk

def process_large_json_parallel(filename):
    # Use all available CPUs minus 1 to leave room for OS
    num_processes = max(1, os.cpu_count() - 1)
    chunk_size = 1000
    
    with mp.Pool(processes=num_processes) as pool:
        with open(filename) as f:
            # Process chunks in parallel
            results = pool.map(process_chunk, read_json_in_chunks(f, chunk_size))
    
    # Merge results from all processes
    final_results = merge_results(results)
    
    # Print results efficiently
    for participant in final_results:
        print(f"\n{participant}:")
        print("Nodes:", end=" ")
        print(*final_results[participant], sep=", ")

if __name__ == '__main__':
    try:
        process_large_json_parallel('ALL_GAM_FIXED.json')
    except Exception as e:
        print(f"Error: {e}")



import pandas as pd
import numpy as np

def find_mixed_rows(csv_file):
    # Read the CSV
    print(f"Reading {csv_file}...")
    df = pd.read_csv(csv_file, index_col=0)
    
    # Print basic info
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Number of participants: {len(df)}")
    print(f"Number of nodes: {len(df.columns)}")
    
    # Check value types
    print("\nValue types in DataFrame:")
    print(df.dtypes.value_counts())
    
    # Check unique values
    unique_vals = np.unique(df.values)
    print(f"\nUnique values in DataFrame: {unique_vals}")
    
    print("\nChecking each row...")
    for index, row in df.iterrows():
        # Get unique values in this row
        row_unique = np.unique(row)
        print(f"\nParticipant {index}: unique values = {row_unique}")
        
        if len(row_unique) > 1:  # If row has more than one unique value
            print(f"\nMIXED ROW FOUND!")
            print(f"Participant: {index}")
            print(f"Values: {row.to_list()}")
            print(f"Number of 0s: {(row == 0).sum()}")
            print(f"Number of 1s: {(row == 1).sum()}")
            print("-" * 50)
            
    # Print summary
    print("\nSummary:")
    print(df.describe())

if __name__ == "__main__":
    try:
        find_mixed_rows("results_matrix.csv")
    except FileNotFoundError:
        print("Error: results_matrix.csv not found!")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())



import json
from collections import defaultdict
import multiprocessing as mp
from itertools import islice
import os
import csv
from datetime import datetime
import pandas as pd

def log_progress(message):
   timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
   print(f"[{timestamp}] {message}", flush=True)

def process_chunk(chunk_data):
   chunk_num, chunk = chunk_data
   log_progress(f"Processing chunk {chunk_num}")
   nodes = defaultdict(set)
   for record in chunk:
       if path := record.get('path'):
           if mapping := path.get('mapping'):
               name = record.get('name')
               if name:
                   nodes[name].update(
                       m['position']['node_id']
                       for m in mapping
                       if 'position' in m and 'node_id' in m['position']
                   )
   return dict(nodes)

def merge_results(results):
   log_progress("Merging results...")
   merged = defaultdict(set)
   for result in results:
       for name, nodes in result.items():
           merged[name].update(nodes)
   return merged

def read_json_in_chunks(file, chunk_size=1000):
   chunk = []
   chunk_num = 0
   for line in file:
       try:
           record = json.loads(line)
           chunk.append(record)
           if len(chunk) == chunk_size:
               chunk_num += 1
               yield (chunk_num, chunk)
               chunk = []
       except json.JSONDecodeError:
           continue
   if chunk:
       chunk_num += 1
       yield (chunk_num, chunk)

def save_to_matrix(final_results, output_prefix):
   log_progress("Creating presence/absence matrix...")
   
   # Get all unique nodes and participants
   all_nodes = set()
   for nodes in final_results.values():
       all_nodes.update(nodes)
   all_nodes = sorted(all_nodes)
   
   # Create matrix as DataFrame
   matrix_data = []
   participants = sorted(final_results.keys())
   
   log_progress(f"Creating matrix with {len(participants)} participants and {len(all_nodes)} nodes")
   
   for participant in participants:
       participant_nodes = final_results[participant]
       row = [1 if node in participant_nodes else 0 for node in all_nodes]
       matrix_data.append(row)
   
   # Create DataFrame
   df = pd.DataFrame(matrix_data, index=participants, columns=all_nodes)
   
   # Save as CSV
   csv_file = f"{output_prefix}_matrix.csv"
   df.to_csv(csv_file)
   log_progress(f"Matrix saved to {csv_file}")
   
   # Save as compressed npz (more efficient for large matrices)
   npz_file = f"{output_prefix}_matrix.npz"
   import numpy as np
   np.savez_compressed(npz_file, 
                      matrix=df.values, 
                      participants=df.index, 
                      nodes=df.columns)
   log_progress(f"Compressed matrix saved to {npz_file}")
   
   return csv_file, npz_file

def process_large_json_parallel(filename, output_prefix="results"):
   start_time = datetime.now()
   log_progress(f"Starting processing of {filename}")
   
   # Use all available CPUs minus 1
   num_processes = max(1, os.cpu_count() - 1)
   chunk_size = 1000
   
   with mp.Pool(processes=num_processes) as pool:
       with open(filename) as f:
           chunks_iterator = read_json_in_chunks(f, chunk_size)
           results = pool.map(process_chunk, chunks_iterator)
   
   final_results = merge_results(results)
   
   # Save raw results
   log_progress("Saving raw results...")
   raw_json = f"{output_prefix}_raw.json"
   with open(raw_json, 'w') as f:
       json.dump({k: list(v) for k, v in final_results.items()}, f, indent=2)
   
   # Save readable format
   log_progress("Saving human-readable format...")
   txt_file = f"{output_prefix}_readable.txt"
   with open(txt_file, 'w') as f:
       for participant, nodes in final_results.items():
           f.write(f"\n{participant}:\n")
           f.write(f"Nodes: {', '.join(str(node) for node in sorted(nodes))}\n")
   
   # Create and save presence/absence matrix
   csv_file, npz_file = save_to_matrix(final_results, output_prefix)
   
   end_time = datetime.now()
   duration = end_time - start_time
   
   log_progress(f"Processing complete! Duration: {duration}")
   log_progress(f"Files created:")
   log_progress(f"1. Raw JSON: {raw_json}")
   log_progress(f"2. Readable text: {txt_file}")
   log_progress(f"3. CSV matrix: {csv_file}")
   log_progress(f"4. Compressed matrix: {npz_file}")

if __name__ == '__main__':
   try:
       process_large_json_parallel('ALL_GAM_FIXED.json')
   except Exception as e:
       log_progress(f"Error: {e}")
       import traceback
       log_progress(traceback.format_exc())



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, 
                             recall_score, f1_score, classification_report, 
                             balanced_accuracy_score, roc_auc_score, average_precision_score, 
                             roc_curve)

# We previously saved the standardized PRS scores, so we load them directly from the GCS path used in the workflow:
prs_scores_gcs = f'{project_bucket}/scores/{target_prs_id}/calculated_scores/score/{target_prs_id}_scores_standardized.csv'
prs_scores_final_df = pd.read_csv(prs_scores_gcs)

# We also previously retrieved the T1D phenotype data from BigQuery.
# To remain consistent and load the same data as before, we will re-run that same query:
workspace_cdr = os.environ["WORKSPACE_CDR"]
T1D_query = f"""
SELECT 
    person_id AS s,
    MAX(
        CASE WHEN condition_source_value IN (
            'E10', 'E10.0', 'E10.1', 'E10.2', 'E10.3', 'E10.4', 'E10.5', 
            'E10.6', 'E10.7', 'E10.8', 'E10.9',
            '250.01', '250.03', '250.11', '250.13', '250.21', '250.23', 
            '250.31', '250.33', '250.41', '250.43', '250.51', '250.53', 
            '250.61', '250.63', '250.71', '250.73', '250.81', '250.83', 
            '250.91', '250.93'
        )
        THEN 1 
        ELSE 0 
        END
    ) AS T1D_status
FROM `{workspace_cdr}.condition_occurrence`
GROUP BY person_id
"""
T1D_phenotype = pd.read_gbq(T1D_query, dialect='standard')

# Recreate merged_scores from the loaded data:
merged_scores = prs_scores_final_df.merge(T1D_phenotype, on='s', how='inner')

# Now we have merged_scores again and can proceed with the logistic regression and threshold adjustment

X = merged_scores[['prs_zscore']]
y = merged_scores['T1D_status']

log_model = LogisticRegression()
log_model.fit(X, y)

y_pred_proba = log_model.predict_proba(X)[:,1]
fpr_arr, tpr_arr, thresholds = roc_curve(y, y_pred_proba)

youden_index = tpr_arr - fpr_arr
best_thresh = thresholds[youden_index.argmax()]

y_pred_adjusted = (y_pred_proba >= best_thresh).astype(int)

acc = accuracy_score(y, y_pred_adjusted)
prec = precision_score(y, y_pred_adjusted, zero_division=0)
rec = recall_score(y, y_pred_adjusted, zero_division=0)
f1 = f1_score(y, y_pred_adjusted, zero_division=0)
balanced_acc = balanced_accuracy_score(y, y_pred_adjusted)
avg_precision = average_precision_score(y, y_pred_proba)
cm = confusion_matrix(y, y_pred_adjusted)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0
auc_val = roc_auc_score(y, y_pred_proba)

print("Threshold:", best_thresh)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("AUC:", auc_val)
print("Balanced Accuracy:", balanced_acc)
print("Specificity:", specificity)
print("PPV:", positive_predictive_value)
print("NPV:", negative_predictive_value)
print("Average Precision:", avg_precision)
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y, y_pred_adjusted, zero_division=0))



from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report, balanced_accuracy_score, roc_auc_score, average_precision_score

# Predictions based on the logistic model
y_pred = log_model.predict(X)
y_pred_proba = log_model.predict_proba(X)[:,1]

# Compute various evaluation metrics
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec = recall_score(y, y_pred, zero_division=0)  # This is also sensitivity
f1 = f1_score(y, y_pred, zero_division=0)
balanced_acc = balanced_accuracy_score(y, y_pred)
avg_precision = average_precision_score(y, y_pred_proba)
auc_val = roc_auc_score(y, y_pred_proba)  # Recalculate in case we need to confirm it's still correct

cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0
positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0

print("Classification Metrics for T1D Prediction Using PRS \n")

print("Accuracy (ACC):", round(acc, 3))

print("\nPrecision (PPV):", round(prec, 3))


print("\nRecall (Sensitivity):", round(rec, 3))


print("\nF1 Score:", round(f1, 3))


print("\nAUC:", round(auc_val, 3))

print("\nSpecificity:", round(specificity, 3))


print("\nPositive Predictive Value (PPV):", round(positive_predictive_value, 3))

print("\nNegative Predictive Value (NPV):", round(negative_predictive_value, 3))


print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:\n")
print(classification_report(y, y_pred, zero_division=0))



from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
                             average_precision_score, roc_curve, classification_report)
import numpy as np

X = merged_scores[['prs_zscore']]
y = merged_scores['T1D_status']

# Split into train/test sets for a fair evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

############################################
# Logistic Regression (Baseline Revisited) #
############################################
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_proba_log = log_model.predict_proba(X_test)[:,1]

# Compute metrics for logistic regression
log_auc = roc_auc_score(y_test, y_proba_log)
log_avg_precision = average_precision_score(y_test, y_proba_log)

print("=== Baseline Logistic Regression ===")
print("ROC-AUC:", round(log_auc, 3))
print("Average Precision:", round(log_avg_precision, 3))

###########################################
# Random Forest Classifier with Tuning    #
###########################################
rf_param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='average_precision', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

y_proba_rf = best_rf.predict_proba(X_test)[:,1]
rf_auc = roc_auc_score(y_test, y_proba_rf)
rf_avg_precision = average_precision_score(y_test, y_proba_rf)

print("\n=== Random Forest (Tuned) ===")
print("Best Params:", rf_grid.best_params_)
print("ROC-AUC:", round(rf_auc, 3))
print("Average Precision:", round(rf_avg_precision, 3))

################################################
# Gradient Boosting Classifier with Tuning     #
################################################
gb_param_grid = {
    'n_estimators': [100, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}
gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_param_grid, cv=3, scoring='average_precision', n_jobs=-1)
gb_grid.fit(X_train, y_train)
best_gb = gb_grid.best_estimator_

y_proba_gb = best_gb.predict_proba(X_test)[:,1]
gb_auc = roc_auc_score(y_test, y_proba_gb)
gb_avg_precision = average_precision_score(y_test, y_proba_gb)

print("\n=== Gradient Boosting (Tuned) ===")
print("Best Params:", gb_grid.best_params_)
print("ROC-AUC:", round(gb_auc, 3))
print("Average Precision:", round(gb_avg_precision, 3))

###########################################
# Summary of Model Comparisons            #
###########################################
print("\n=== Model Comparison ===")
print("Model                |   AUC    |   Avg Precision")
print("-------------------------------------------------")
print(f"Logistic Regression  |  {log_auc:.3f}  |   {log_avg_precision:.3f}")
print(f"Random Forest (tuned)|  {rf_auc:.3f}  |   {rf_avg_precision:.3f}")
print(f"Gradient Boosting    |  {gb_auc:.3f}  |   {gb_avg_precision:.3f}")



import os
import subprocess

def run_cmd(cmd, description):
    print(f"[COMMAND]: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    for line in proc.stdout:
        line_stripped = line.strip()
        output.append(line_stripped)
        print(line_stripped)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Error running {description}, command: {' '.join(cmd)}")
    return output

# ---------------------------------------------------------------------------
# Validate environment
# ---------------------------------------------------------------------------
project_id = os.getenv("GOOGLE_PROJECT")
if project_id is None:
    raise ValueError("GOOGLE_PROJECT environment variable is not set!")

# ---------------------------------------------------------------------------
# Set up paths and parameters
# Use the ACAF threshold srWGS callset (version 7.1) for chr6:
# gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/vcf/acaf_threshold.chr6.vcf.bgz
# and the index acaf_threshold.chr6.vcf.bgz.tbi
remote_vcf = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/vcf/acaf_threshold.chr6.vcf.bgz"
remote_tbi = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/acaf_threshold_v7.1/vcf/acaf_threshold.chr6.vcf.bgz.tbi"

local_vcf = "acaf_threshold.chr6.vcf.bgz"
local_tbi = "acaf_threshold.chr6.vcf.bgz.tbi"

region = "chr6:31283525-31306849"

# Reference FASTA and index (hg38)
reference_fasta = "Homo_sapiens_assembly38.fasta"
reference_fai = "Homo_sapiens_assembly38.fasta.fai"

# Output FASTA filename will be determined after we find the first sample name
output_fasta = "consensus_chr6_31283525_31306849.fasta"

# ---------------------------------------------------------------------------
# Download reference FASTA and index if not present
# ---------------------------------------------------------------------------
if not os.path.exists(reference_fasta):
    run_cmd(["gsutil", "cp",
             "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta",
             reference_fasta],
            "downloading reference FASTA")

if not os.path.exists(reference_fai):
    run_cmd(["gsutil", "cp",
             "gs://gcp-public-data--broad-references/hg38/v0/Homo_sapiens_assembly38.fasta.fai",
             reference_fai],
            "downloading reference FASTA index")

# ---------------------------------------------------------------------------
# Download the VCF and its index locally
# ---------------------------------------------------------------------------
run_cmd(["gsutil", "-u", project_id, "cp", remote_vcf, local_vcf], "downloading chr6 VCF")
run_cmd(["gsutil", "-u", project_id, "cp", remote_tbi, local_tbi], "downloading chr6 VCF index")

# ---------------------------------------------------------------------------
# Determine the first sample name in the VCF
# ---------------------------------------------------------------------------
samples = run_cmd(["bcftools", "query", "-l", local_vcf], "listing samples")
if not samples:
    raise RuntimeError("No samples found in the VCF.")
first_sample = samples[0]
print(f"[INFO]: First sample in VCF is {first_sample}")

# Update output filename to include the sample name
output_fasta = f"{first_sample}_chr6_31283525_31306849.fasta"

# ---------------------------------------------------------------------------
# Generate consensus FASTA for the first sample and region
# ---------------------------------------------------------------------------
consensus_cmd = [
    "bash", "-c",
    f"bcftools view -r {region} {local_vcf} | "
    f"bcftools consensus -f {reference_fasta} -s {first_sample} > {output_fasta}"
]

run_cmd(consensus_cmd, "generating consensus FASTA")

print(f"[INFO]: FASTA for {first_sample} in {region} created: {output_fasta}")

