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

def download_prs_files(prs_id, harmonized_url, original_url):
    harmonized_local = f'{prs_id}_hmPOS_GRCh38.txt.gz'
    original_local = f'{prs_id}.txt.gz'

    print("Downloading harmonized PRS file from:", harmonized_url)
    response = requests.get(harmonized_url)
    with open(harmonized_local, 'wb') as out_file:
        out_file.write(response.content)
    print("Harmonized PRS weight file downloaded.\n")

    print("Downloading original PRS file for comparison...")
    !wget -O {original_local} {original_url}
    print("Original PRS weight file downloaded.\n")

    print("Listing local files for verification...")
    !ls
    print("\nDecompressing harmonized PRS file for inspection...")
    !gunzip -f {harmonized_local}
    return harmonized_local.replace('.gz', ''), original_local

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

    print("Adding Hail-compatible columns: contig, position, variant_id")
    df['chr'] = df['chr'].astype(str)
    df['contig'] = df['chr'].apply(lambda c: c if c.startswith('chr') else f'chr{c}')
    df['position'] = df['bp']
    df['variant_id'] = df.apply(lambda row: f"{row['contig']}:{row['position']}", axis=1)
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

    essential = ["variant_id", "weight", "contig", "position", "effect_allele"]
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

target_prs_id = 'PGS000001'
harmonized_url = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000001/ScoringFiles/Harmonized/PGS000001_hmPOS_GRCh38.txt.gz'
original_prs_url = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000001/ScoringFiles/PGS000001.txt.gz'
harmonized_file, _ = download_prs_files(target_prs_id, harmonized_url, original_prs_url)
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

print("Visualizing distributions...")
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
sns.histplot(data=prs_scores_final_df, x='prs_zscore', bins=50)
plt.title('T1D PRS Z-Score Distribution')
plt.xlabel('PRS Z-score')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
sns.boxplot(data=prs_scores_final_df, x='risk_category', y='prs_zscore')
plt.title('PRS Z-score by Risk Category')
plt.xticks(rotation=45)
plt.tight_layout()

print("Z-score stats:")
print(prs_scores_final_df['prs_zscore'].describe(), "\n")
print("Counts in each PRS risk category:")
print(prs_scores_final_df['risk_category'].value_counts().sort_index(), "\n")

standardized_scores_path = f'{project_bucket}/scores/{target_prs_id}/calculated_scores/score/{target_prs_id}_scores_standardized.csv'
print("Saving standardized PRS scores (with z-scores and categories) to:", standardized_scores_path)
prs_scores_final_df.to_csv(standardized_scores_path, index=False)
print("Standardized PRS scores saved.\n")

plt.show()

print("Evaluating PRS against T1D phenotype...")
workspace_cdr = os.environ["WORKSPACE_CDR"]

# These codes are not correct for T1D. They need to be switched
T1D_query = f"""
SELECT 
    person_id as s,
    MAX(CASE WHEN condition_source_value IN ('I25.1', 'I25.10', 'I25.11', 'I25.118', 'I25.119',
                                           'I25.2', 'I25.3', 'I25.41', 'I25.42', 'I25.5', 'I25.6',
                                           'I25.89', 'I25.9', '414.00', '414.01', '414.0', '414')
             THEN 1 ELSE 0 END) as T1D_status
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
plt.plot(fpr_arr, tpr_arr, color='blue', lw=2, label=f'ROC (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('T1D PRS ROC Curve')
plt.legend(loc="lower right")

merged_scores['prs_quintile'] = pd.qcut(merged_scores['prs_zscore'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
quintile_stats = merged_scores.groupby('prs_quintile').agg({'T1D_status': ['count', 'mean']}).reset_index()

print("=== T1D Association Summary ===")
print("ROC-AUC:", round(auc_val, 3))
print("\nT1D prevalence by PRS quintile:")
print(quintile_stats, "\n")
plt.show()

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
             ha='center', va='bottom')

plt.title('T1D Prevalence in Top vs Bottom 5% of PRS Distribution', pad=20)
plt.ylabel('T1D Prevalence (%)')
plt.tight_layout()

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
plt.show()

job_end_time = datetime.datetime.now()
print("********** JOB COMPLETED **********")
print("Completion date:", job_end_time.strftime("%Y-%m-%d"))
print("Completion time:", job_end_time.strftime("%H:%M:%S"))
print("All steps finished successfully.")
print("***********************************\n")
