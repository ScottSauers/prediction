#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install AoUPRS


# In[2]:


# Record start time
import datetime
start_time = datetime.datetime.now()
print("Start date:", start_time.strftime("%Y-%m-%d"))
print("Start time:", start_time.strftime("%H:%M:%S"))

# Import necessary modules
import os
import hail as hl
import pandas as pd
import gcsfs
import AoUPRS

# Define bucket
bucket = os.getenv("WORKSPACE_BUCKET")
print("Bucket:", bucket)

# Initialize Hail
hl.init(tmp_dir=f'{bucket}/hail_temp/', default_reference='GRCh38')

# Read Hail VDS
vds_srwgs_path = 'gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/vds/hail.vds'
vds = hl.vds.read_vds(vds_srwgs_path)
print("Number of samples in VDS:", vds.n_samples())

# Drop flagged samples
flagged_samples_path = "gs://fc-aou-datasets-controlled/v7/wgs/short_read/snpindel/aux/relatedness/relatedness_flagged_samples.tsv"
flagged_samples = hl.import_table(flagged_samples_path, key='sample_id')
vds_no_flag = hl.vds.filter_samples(vds, flagged_samples, keep=False)
print("Number of samples after removing flagged samples:", vds_no_flag.n_samples())

# Define the sample intended for PRS calculation
# Get list of person_ids with WGS and EHR data
import os

query = f"""
SELECT person_id
FROM `{os.environ["WORKSPACE_CDR"]}.person`
WHERE person_id IN (
    SELECT DISTINCT person_id
    FROM `{os.environ["WORKSPACE_CDR"]}.cb_search_person`
    WHERE has_ehr_data = 1
    AND has_whole_genome_variant = 1
)
"""

# Execute the query and get the person_ids
sample_ids_df = pd.read_gbq(query, dialect='standard')
print(f"Number of samples with WGS and EHR data: {sample_ids_df['person_id'].nunique()}")

# Save sample IDs to a CSV file in the bucket
sample_ids_path = f'{bucket}/prs_calculator_tutorial/people_with_WGS_EHR_ids.csv'
sample_ids_df.to_csv(sample_ids_path, index=False)

# Import sample IDs into Hail Table
sample_needed_ht = hl.import_table(sample_ids_path, delimiter=',', key='person_id')
# Filter samples
vds_subset = hl.vds.filter_samples(vds_no_flag, sample_needed_ht, keep=True)
print("Number of samples after filtering for WGS and EHR data:", vds_subset.n_samples())


# In[3]:


# Prepare PRS Weight Table using AoUPRS

# Download PRS weight file for PGS000746 from the PGS Catalog
import requests

prs_identifier = 'PGS000746'
pgs_weight_url = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000746/ScoringFiles/Harmonized/PGS000746_hmPOS_GRCh38.txt.gz'
pgs_weight_local_path = 'PGS000746_hmPOS_GRCh38.txt.gz'

# Download the PRS weight file
print("Downloading PRS weight file...")
response = requests.get(pgs_weight_url)
with open(pgs_weight_local_path, 'wb') as f:
    f.write(response.content)
print("Download completed.")


# In[4]:


get_ipython().system('ls')


# In[5]:


# Set the PRS identifier
prs_identifier = 'PGS000746'

# Correct the URL to use HTTPS protocol
pgs_weight_url = 'https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS000746/ScoringFiles/PGS000746.txt.gz'
pgs_weight_local_path = 'PGS000746.txt.gz'

# Download the PRS weight file using wget
print("Downloading PRS weight file...")
get_ipython().system('wget -O {pgs_weight_local_path} {pgs_weight_url}')
print("Download completed.")


# In[6]:


get_ipython().system('ls')


# In[7]:


get_ipython().system('gunzip PGS000746_hmPOS_GRCh38.txt.gz -y')


# In[8]:


get_ipython().system('head -n 25  PGS000746_hmPOS_GRCh38.txt')


# In[9]:


# Read the PRS weight table into a pandas DataFrame
prs_df = pd.read_csv('PGS000746_hmPOS_GRCh38.txt', sep='\t', comment='#')
print(f"Number of variants in the PRS weight file: {len(prs_df)}")

# Prepare the PRS weight table using harmonized positions
# Rename columns to match expected names
prs_df = prs_df.rename(columns={
    'hm_chr': 'chr',
    'hm_pos': 'bp',
    'effect_allele': 'effect_allele',
    'other_allele': 'noneffect_allele',
    'effect_weight': 'weight'
})

# Keep only the necessary columns
prs_df = prs_df[['chr', 'bp', 'effect_allele', 'noneffect_allele', 'weight']]

# Convert chromosome column to string if not already
prs_df['chr'] = prs_df['chr'].astype(str)

# Save the prepared PRS weight table to a CSV file
prs_weight_table_path = 'PGS000746_prepared_weight_table.csv'
prs_df.to_csv(prs_weight_table_path, index=False)
print("PRS weight table saved locally.")


# In[ ]:





# In[10]:


# Define the output path relative to your bucket
output_prs_weight_path = 'AoUPRS/PGS000746/weight_table/PGS000746_weight_table.csv'

# Copy the local prepared weight table to your bucket
get_ipython().system('gsutil cp {prs_weight_table_path} {bucket}/{output_prs_weight_path}')

# Prepare the PRS table using AoUPRS function
AoUPRS.prepare_prs_table(
    output_prs_weight_path,  # Input path (relative to bucket)
    output_prs_weight_path,  # Output path (relative to bucket)
    bucket=bucket
)
print("PRS weight table prepared and saved to:", f'{bucket}/{output_prs_weight_path}')


# In[11]:


# Now proceed to calculate the PRS using AoUPRS.calculate_prs_vds

output_path = f'AoUPRS/PGS000746/calculated_scores'

AoUPRS.calculate_prs_vds(
    vds=vds_subset,
    prs_identifier=prs_identifier,
    pgs_weight_path=output_prs_weight_path,
    output_path=output_path,
    bucket=bucket,
    save_found_variants=True
)

print("PRS calculation completed.")

# Record end time
end_time = datetime.datetime.now()
print("End date:", end_time.strftime("%Y-%m-%d"))
print("End time:", end_time.strftime("%H:%M:%S"))


# In[4]:


import os
import pandas as pd

# Define bucket
bucket = os.getenv("WORKSPACE_BUCKET")

# Define paths
scores_path = f'{bucket}/AoUPRS/PGS000746/calculated_scores/score/PGS000746_scores.csv'
variants_path = f'{bucket}/AoUPRS/PGS000746/calculated_scores/score/PGS000746_found_in_aou.csv'

# Read the files
print("Reading scores file...")
scores_df = pd.read_csv(scores_path)
print("\nReading variants file...")
variants_df = pd.read_csv(variants_path)

print("\nScore statistics:")
print(scores_df['sum_weights'].describe())

print("\nConfirming number of variants per sample:")
print(scores_df['N_variants'].value_counts())


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate standardized scores and percentiles
scores_df['prs_zscore'] = (scores_df['sum_weights'] - scores_df['sum_weights'].mean()) / scores_df['sum_weights'].std()
scores_df['prs_percentile'] = scores_df['prs_zscore'].rank(pct=True) * 100

# Add risk categories
scores_df['risk_category'] = pd.qcut(scores_df['prs_zscore'], 
                                   q=5, 
                                   labels=['Very Low', 'Low', 'Average', 'High', 'Very High'])

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Distribution of Z-scores
plt.subplot(2, 1, 1)
sns.histplot(data=scores_df, x='prs_zscore', bins=50)
plt.title('Distribution of CAD PRS Z-scores')
plt.xlabel('PRS Z-score')
plt.ylabel('Count')

# Plot 2: Box plot by risk category
plt.subplot(2, 1, 2)
sns.boxplot(data=scores_df, x='risk_category', y='prs_zscore')
plt.title('PRS Z-scores by Risk Category')
plt.xticks(rotation=45)
plt.tight_layout()

# Show summary statistics
print("=== PRS Distribution Summary ===")
print("\nZ-score statistics:")
print(scores_df['prs_zscore'].describe())

print("\nDistribution across risk categories:")
print(scores_df['risk_category'].value_counts().sort_index())

# Save the standardized scores
standardized_scores_path = f'{bucket}/AoUPRS/PGS000746/calculated_scores/score/PGS000746_scores_standardized.csv'
scores_df.to_csv(standardized_scores_path, index=False)
print(f"\nStandardized scores saved to: {standardized_scores_path}")

plt.show()


# In[7]:


import os
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# First, let's get the CAD phenotype data from AoU
# This query gets CAD status using ICD codes
query = f"""
SELECT 
    person_id as s,
    MAX(CASE WHEN condition_source_value IN ('I25.1', 'I25.10', 'I25.11', 'I25.118', 'I25.119',
                                           'I25.2', 'I25.3', 'I25.41', 'I25.42', 'I25.5', 'I25.6',
                                           'I25.89', 'I25.9', '414.00', '414.01', '414.0', '414') THEN 1 ELSE 0 END) as cad_status
FROM `{os.environ["WORKSPACE_CDR"]}.condition_occurrence`
GROUP BY person_id
"""

# Execute the query and get phenotype data
print("Fetching CAD phenotype data...")
phenotype_df = pd.read_gbq(query, dialect='standard')

# Merge with PRS scores
merged_df = scores_df.merge(phenotype_df, on='s', how='inner')

# Calculate ROC-AUC
auc = roc_auc_score(merged_df['cad_status'], merged_df['prs_zscore'])
fpr, tpr, _ = roc_curve(merged_df['cad_status'], merged_df['prs_zscore'])

# Create ROC curve plot
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CAD PRS')
plt.legend(loc="lower right")

# Calculate odds ratios across PRS quintiles
merged_df['prs_quintile'] = pd.qcut(merged_df['prs_zscore'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Calculate CAD prevalence by quintile
quintile_stats = merged_df.groupby('prs_quintile').agg({
    'cad_status': ['count', 'mean']
}).reset_index()

print("\n=== CAD PRS Performance ===")
print(f"ROC-AUC: {auc:.3f}")
print("\nCAD prevalence by PRS quintile:")
print(quintile_stats)

plt.show()


# In[11]:


import scipy.stats as stats
import numpy as np

# Verify CAD cases
print("=== CAD Case Verification ===")
print(f"Total samples: {len(merged_df)}")
print(f"Number of CAD cases: {merged_df['cad_status'].sum()}")
print(f"CAD prevalence: {(merged_df['cad_status'].mean()*100):.2f}%")

# Calculate p-value using logistic regression
from sklearn.linear_model import LogisticRegression
X = merged_df['prs_zscore'].values.reshape(-1, 1)
y = merged_df['cad_status'].values
log_reg = LogisticRegression()
log_reg.fit(X, y)
p_value = stats.chi2.sf(log_reg.score(X, y) * len(y), df=1)

print("\n=== Statistical Testing ===")
print(f"Logistic Regression p-value: {p_value:.2e}")

# Get 5th and 95th percentile groups
threshold_top = np.percentile(merged_df['prs_zscore'], 95)
threshold_bottom = np.percentile(merged_df['prs_zscore'], 5)

# Calculate stats for each group
bottom_5 = merged_df[merged_df['prs_zscore'] <= threshold_bottom]
top_5 = merged_df[merged_df['prs_zscore'] >= threshold_top]

results = {
    'Bottom 5%': {
        'prevalence': bottom_5['cad_status'].mean() * 100,
        'n_total': len(bottom_5),
        'n_cases': bottom_5['cad_status'].sum()
    },
    'Top 5%': {
        'prevalence': top_5['cad_status'].mean() * 100,
        'n_total': len(top_5),
        'n_cases': top_5['cad_status'].sum()
    }
}

# Create visualization
plt.figure(figsize=(10, 6))
groups = ['Bottom 5%', 'Top 5%']
prevalences = [results[k]['prevalence'] for k in groups]

bars = plt.bar(groups, prevalences, 
               color=['lightblue', 'darkblue'])

# Add value labels on top of bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    group = groups[i]
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'CAD cases={results[group]["n_cases"]:,}',
             ha='center', va='bottom')

plt.title('CAD Prevalence: Top 5% vs Bottom 5% PRS', pad=20)
plt.ylabel('CAD Prevalence (%)')

# Calculate odds ratio
odds_ratio = (results['Top 5%']['n_cases'] / (results['Top 5%']['n_total'] - results['Top 5%']['n_cases'])) / \
             (results['Bottom 5%']['n_cases'] / (results['Bottom 5%']['n_total'] - results['Bottom 5%']['n_cases']))

print("\n=== Group Comparisons ===")
for name, data in results.items():
    print(f"\n{name}:")
    print(f"  Prevalence: {data['prevalence']:.2f}%")
    print(f"  Total individuals: {data['n_total']:,}")
    print(f"  CAD cases: {data['n_cases']:,}")

print(f"\nOdds Ratio (Top 5% vs Bottom 5%): {odds_ratio:.2f}")

plt.tight_layout()
plt.show()
