import streamlit as st
import util.constants.filepaths
import pandas as pd
import zipfile
import io
import os
from streamlit_option_menu import option_menu
from util.functions.header import header
from util.functions.filter_dataframe import filter_dataframe

header(__file__.split('/')[-1].split('.')[0])
st.header("Browse and Download")
st.write("Browse complex structure data in the searchable and filterable table below. Use checkboxes in the table to select complexes for model structure download. " \
"" \
"To download the data table itself, move cursor to the top right corner of the table and click on the download icon.")

# Add custom CSS to style the dataframe header
st.markdown("""
<style>
    /* Style the dataframe header to match background */
    [data-testid="stDataFrameResizable"] thead tr th {
        background-color: #FDD1D7 !important;
        color: #000000 !important;
    }
    
    /* Also style the column headers */
    [data-testid="stDataFrameResizable"] div[data-testid="stDataFrameColHeader"] {
        background-color: #FDD1D7 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the data first
df = pd.read_csv(util.constants.filepaths.MAIN_DATASET_FILE)
# add Selected column
df.insert(0, 'Selected', False)

# Initialize session state for selections if not exists
if 'selected_rows' not in st.session_state:
    st.session_state['selected_rows'] = set()

# Create a filtered dataframe
filtered_df = filter_dataframe(df)

# Apply stored selections from session state
for idx in filtered_df.index:
    if idx in st.session_state['selected_rows']:
        filtered_df.loc[idx, 'Selected'] = True

# Create a display dataframe with URL columns for UniProt IDs
display_df = filtered_df.copy()

# Function to add URL columns for comma-separated UniProt IDs
def add_uniprot_url_columns(df, id_column, prefix):
    # Create URL columns for each position consistently
    url_columns = []
    for idx, row in df.iterrows():
        if pd.notna(row[id_column]) and row[id_column] != '':
            ids = str(row[id_column]).split(',')
            # Create URL columns for each ID
            for i, id_val in enumerate(ids):
                col_name = f'{prefix}_URL_{i+1}'
                # Create column if it doesn't exist
                if col_name not in df.columns:
                    df[col_name] = ""
                    url_columns.append(col_name)
                # Set the value for this specific row
                df.at[idx, col_name] = f"https://uniprot.org/uniprotkb/{id_val.strip()}/entry"
    
    # Return the list of all URL column names created
    return [col for col in df.columns if col.startswith(f'{prefix}_URL')]

# Create URL columns for E2 and E3 UniProt IDs
e2_url_columns = add_uniprot_url_columns(display_df, 'E2_uniprot', 'E2')
e3_url_columns = add_uniprot_url_columns(display_df, 'E3_uniprot', 'E3')

# Remove the original UniProt columns and rename URL columns to replace them
if 'E2_uniprot' in display_df.columns:
    display_df.drop('E2_uniprot', axis=1, inplace=True)
if 'E3_uniprot' in display_df.columns:
    display_df.drop('E3_uniprot', axis=1, inplace=True)

# Rename the first URL column for each protein type to replace the original columns
if e2_url_columns:
    display_df.rename(columns={e2_url_columns[0]: 'E2_uniprot'}, inplace=True)
    e2_url_columns[0] = 'E2_uniprot'  # Update the list to reflect the rename

if e3_url_columns:
    display_df.rename(columns={e3_url_columns[0]: 'E3_uniprot'}, inplace=True)
    e3_url_columns[0] = 'E3_uniprot'  # Update the list to reflect the rename

# Reorder columns to place E2_uniprot and E3_uniprot in 4th and 5th positions
cols = display_df.columns.tolist()

# Remove the UniProt columns from their current positions
uniprot_cols = []
if 'E2_uniprot' in cols:
    cols.remove('E2_uniprot')
    uniprot_cols.append('E2_uniprot')
if 'E3_uniprot' in cols:
    cols.remove('E3_uniprot')
    uniprot_cols.append('E3_uniprot')

# Insert the UniProt columns at positions 3 and 4 (0-indexed, so 4th and 5th position)
for i, col in enumerate(uniprot_cols):
    cols.insert(4 + i, col)

# Reorder the dataframe
display_df = display_df[cols]

# Apply gene filter if it exists in session state (BEFORE displaying dataframe)
if 'gene_filter' in st.session_state:
    filter_info = st.session_state['gene_filter']
    
    # Show filter info with option to clear
    col_filter, col_clear = st.columns([4, 1])
    with col_filter:
        st.info(f"Filtered by {filter_info['column']}: **{filter_info['value']}**")
    with col_clear:
        if st.button("Clear Filter", key="clear_gene_filter"):
            del st.session_state['gene_filter']
            st.rerun()
    
    # Apply the filter to display_df
    display_df = display_df[display_df[filter_info['column']] == filter_info['value']]

# Initialize or update the display dataframe based on select all triggers
if 'select_all_trigger' in st.session_state and st.session_state['select_all_trigger']:
    # Add all visible rows to selected_rows
    for idx in display_df.index:
        st.session_state['selected_rows'].add(idx)
    display_df['Selected'] = True
    st.session_state['select_all_trigger'] = False
    st.rerun()  # Force rerun to update display
    
if 'deselect_all_trigger' in st.session_state and st.session_state['deselect_all_trigger']:
    # Remove all visible rows from selected_rows
    for idx in display_df.index:
        st.session_state['selected_rows'].discard(idx)
    display_df['Selected'] = False
    st.session_state['deselect_all_trigger'] = False
    st.rerun()  # Force rerun to update display

# Apply selections to display_df for rendering
for idx in display_df.index:
    if idx in st.session_state['selected_rows']:
        display_df.loc[idx, 'Selected'] = True

def download_selected_complexes():
    """Download selected complex PDB files as a zip"""
    # Get selected complexes from session state and original df
    selected_indices = st.session_state['selected_rows']
    if not selected_indices:
        st.warning("No complexes selected. Please select at least one complex from the table.")
        return
    
    # Get complex names from the original filtered dataframe using indices
    selected_complexes = filtered_df.loc[list(selected_indices), 'Complex'].tolist()
    
    # Create a progress bar and status text
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text(f"Preparing to download {len(selected_complexes)} complex(es)...")
    
    # Create a zip file in memory
    zip_buffer = io.BytesIO()
    missing_files = []
    found_files = 0
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, complex_name in enumerate(selected_complexes):
            # Update progress
            progress = (i + 1) / len(selected_complexes)
            progress_bar.progress(progress)
            progress_text.text(f"Processing {i + 1}/{len(selected_complexes)}: {complex_name}")
            
            pdb_path = f"{util.constants.filepaths.PDB_FILES}/{complex_name}.pdb"
            if os.path.exists(pdb_path):
                # Add the PDB file to the zip
                with open(pdb_path, 'rb') as f:
                    zip_file.writestr(f"{complex_name}.pdb", f.read())
                found_files += 1
            else:
                missing_files.append(complex_name)
    
    # Clear progress indicators
    progress_bar.empty()
    progress_text.empty()
    
    # Show warnings for missing files
    if missing_files:
        st.warning(f"Structure files not found for: {', '.join(missing_files)}")
        
    # If no files were found, show an error and return
    if found_files == 0:
        st.error("No structure files could be found for the selected complexes.")
        return
    
    # Reset buffer position
    zip_buffer.seek(0)
    
    # Create the download button
    st.download_button(
        label=f"Download {found_files} PDB Files",
        data=zip_buffer,
        file_name="selected_complexes.zip",
        mime="application/zip",
        key="download_pdb_files"
    )
    
    st.success(f"{found_files} complex PDB files prepared for download.")

# Buttons above dataframe
col1, col2, col3, col4 = st.columns([2, 0.8, 0.9, 4.8])
with col1:
    if st.button("Download Selected Complexes"):
        download_selected_complexes()
with col2:
    if st.button("Select All", key="select_all_btn"):
        st.session_state['select_all_trigger'] = True
        st.rerun()
with col3:
    if st.button("Deselect All", key="deselect_all_btn"):
        st.session_state['deselect_all_trigger'] = True
        st.rerun()

# Create column config for UniProt URL columns
column_config = {
    "Selected": st.column_config.CheckboxColumn()
}

# Add configuration for the renamed primary UniProt columns
if 'E2_uniprot' in e2_url_columns:
    column_config['E2_uniprot'] = st.column_config.LinkColumn(
        "E2_uniprot",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E2 protein on UniProt"
    )

if 'E3_uniprot' in e3_url_columns:
    column_config['E3_uniprot'] = st.column_config.LinkColumn(
        "E3_uniprot",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E3 protein on UniProt"
    )

# Add configuration for any additional URL columns (for multiple IDs)
for col in e2_url_columns[1:]:  # Skip the first one as it's now renamed to E2_uniprot
    num = col.split('_')[-1]
    column_config[col] = st.column_config.LinkColumn(
        f"E2 UniProt_{num}",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E2 protein on UniProt"
    )

for col in e3_url_columns[1:]:  # Skip the first one as it's now renamed to E3_uniprot
    num = col.split('_')[-1]
    column_config[col] = st.column_config.LinkColumn(
        f"E3 UniProt_{num}",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E3 protein on UniProt"
    )

# Display the dataframe
event=st.dataframe(
    display_df,
    column_config=column_config,
    hide_index=True,
    width='stretch',
    on_select="rerun",
    selection_mode="single-cell"
)

# Handle gene name filtering based on cell selection
if event.selection and 'cells' in event.selection and len(event.selection['cells']) > 0:
    selected_cell = event.selection['cells'][0]
    row_idx = selected_cell[0]  # First element is row index
    col_name = selected_cell[1]  # Second element is already the column name
    
    if col_name == 'Selected':
        # Toggle the Selected checkbox for this row
        actual_row_idx = display_df.index[row_idx]
        current_value = actual_row_idx in st.session_state['selected_rows']
        
        if current_value:
            st.session_state['selected_rows'].discard(actual_row_idx)
        else:
            st.session_state['selected_rows'].add(actual_row_idx)
        st.rerun()
    elif col_name in ['E3_gene_name', 'E2_gene_name']:
        # Get the gene name from the selected cell
        gene_name = display_df.iloc[row_idx][col_name]
        
        if pd.notna(gene_name) and gene_name != '':
            # Store filter in session state
            st.session_state['gene_filter'] = {
                'column': col_name,
                'value': gene_name
            }
            st.rerun()




