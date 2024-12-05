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


# In[ ]:


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

