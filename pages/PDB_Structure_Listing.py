import streamlit as st
import util.constants.filepaths
import pandas as pd
from util.functions.header import header
from util.functions.filter_dataframe import filter_dataframe
from util.functions.get_latest_pdb_table import get_latest_pdb_table
from pathlib import Path
import glob

header(__file__.split('/')[-1].split('.')[0])

st.header('Experimental Structures in PDB')

st.write('This is a collection of experimental structures from the Protein Data Bank (PDB) that include human E2 enzymes and/or E3 ligases.')

date_time, filename = get_latest_pdb_table(util.constants.filepaths.PDB_STRUCTURE_TABLE_FOLDER)

st.write(f"Updated: {date_time}")

# Load the latest PDB structure table
if filename is not None:
    df = pd.read_csv(filename)

# Filter the original dataframe (only call this once)
filtered_df = filter_dataframe(df)

# Create a new display dataframe
display_df = filtered_df.copy()

# Convert ubiquitin_in_structure to Yes/No
if 'ubiquitin_in_structure' in display_df.columns:
    display_df['ubiquitin_in_structure'] = display_df['ubiquitin_in_structure'].apply(
        lambda x: "Yes" if pd.notna(x) and str(x).strip() not in ['', '[]', 'nan'] else "No"
    )

# Add URL columns for PDB and UniProt entries
display_df['PDB_URL'] = display_df['PDB_ID'].apply(
    lambda x: f"https://www.rcsb.org/structure/{x}" if pd.notna(x) else ""
)

# Function to add URL columns for comma-separated UniProt IDs
def add_url_columns(df, id_column, prefix):
    # Create URL columns for each position consistently
    for idx, row in df.iterrows():
        if pd.notna(row[id_column]):
            ids = row[id_column].split(',')
            # Create URL columns for each ID
            for i, id_val in enumerate(ids):
                col_name = f'{prefix}_URL_{i+1}'
                # Create column if it doesn't exist
                if col_name not in df.columns:
                    df[col_name] = ""
                # Set the value for this specific row
                df.at[idx, col_name] = f"https://uniprot.org/uniprotkb/{id_val.strip()}/entry"
    
    # Return the list of all URL column names created
    return [col for col in df.columns if col.startswith(f'{prefix}_URL')]

# Create URL columns for each protein type
e2_url_columns = add_url_columns(display_df, 'E2_UniProtIDs', 'E2')
e3_url_columns = add_url_columns(display_df, 'E3_UniProtIDs', 'E3')

# Reorder columns to put URLs first
cols = display_df.columns.tolist()

# Remove URL columns and ID columns since we'll place them in a specific order
url_columns_to_remove = ['PDB_URL'] + e2_url_columns + e3_url_columns
id_columns_to_remove = ['PDB_ID', 'E2_UniProtIDs', 'E3_UniProtIDs', 'Ub_UniProtIDs']

for col in url_columns_to_remove + id_columns_to_remove:
    if col in cols:
        cols.remove(col)

# Extract the columns to move
special_columns = ['E2_Gene_Names', 'E3_Gene_Names', 'ubiquitin_in_structure']

# Remove these columns from cols if they exist
for col in special_columns:
    if col in cols:
        cols.remove(col)

# Arrange columns with URLs first, special columns next, then the rest organized by type
display_df = display_df[['PDB_URL'] + special_columns + e2_url_columns + e3_url_columns + cols]


# Create column config for dynamic URL columns
column_config = {
    "PDB_URL": st.column_config.LinkColumn(
        "PDB ID",
        display_text="https://www.rcsb.org/structure/(.*)",
        help="Click to view structure on RCSB website"
    ),
    "ubiquitin_in_structure": st.column_config.Column(
        "Ubiquitin_in_structure",
        help="Indicates if ubiquitin is present in the structure",
        width="medium"
    )
}

# Add configuration for all dynamic URL columns
for col in e2_url_columns:
    num = col.split('_')[-1]
    column_config[col] = st.column_config.LinkColumn(
        f"E2 UniProt_{num}",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E2 protein on UniProt"
    )

for col in e3_url_columns:
    num = col.split('_')[-1]
    column_config[col] = st.column_config.LinkColumn(
        f"E3 UniProt_{num}",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E3 protein on UniProt"
    )



# Calculate list of all columns to disable
disabled_cols = list(df.columns) + ['PDB_URL'] + e2_url_columns + e3_url_columns

# Configure all non-link columns
for col in display_df.columns:
    if col not in column_config:
        column_config[col] = st.column_config.Column(
            col,
            help=f"{col} information",
            width="auto"
        )

# Display the dataframe with all URL columns
st.data_editor(
    display_df,
    disabled=disabled_cols,  # Make all columns disabled
    hide_index=True,
    column_config=column_config,
    use_container_width=True
)
