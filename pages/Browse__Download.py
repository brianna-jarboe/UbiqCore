import streamlit as st
import util.constants.filepaths
import pandas as pd
import zipfile
import io
import os
import hashlib
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

# Cache the data loading to prevent full reloads and preserve scroll position where possible
@st.cache_data
def load_data():
    return pd.read_csv(util.constants.filepaths.MAIN_DATASET_FILE)

# Load the data first
df = load_data().copy() # Work on a copy to avoid mutating cached data
# add Selected column
df.insert(0, 'Selected', False)

# Initialize session state for selections if not exists
if 'selected_rows' not in st.session_state:
    st.session_state['selected_rows'] = set()

# Create a filtered dataframe
filtered_df = filter_dataframe(df)

# Logic to determine if we need to rebuild the display dataframe.
# We want to AVOID rebuilding local display_df if the user just clicked a checkbox,
# as passing a new dataframe object to data_editor causes the scroll reset.
# We ONLY rebuild if:
# 1. The filtered data content changed (search/filter applied)
# 2. A bulk action (Select/Deselect All) was triggered
# 3. It's the first load

# Create a signature for the current filtered data view
current_indices = filtered_df.index.tolist()
# Simple signature based on indices and length
view_signature = f"{len(current_indices)}-{hash(tuple(current_indices))}"

# Check triggers
bulk_action = st.session_state.get('select_all_trigger', False) or \
              st.session_state.get('deselect_all_trigger', False)
              
view_changed = st.session_state.get('last_view_signature') != view_signature
cache_missing = 'cached_display_df' not in st.session_state

should_rebuild_display = bulk_action or view_changed or cache_missing

# Process Select All / Deselect All logic BEFORE rebuilding
if st.session_state.get('select_all_trigger'):
    for idx in filtered_df.index:
        st.session_state['selected_rows'].add(idx)
    st.session_state['select_all_trigger'] = False
    # Clear editor state to force UI update
    if "browse_table" in st.session_state:
        del st.session_state["browse_table"]

if st.session_state.get('deselect_all_trigger'):
    for idx in filtered_df.index:
        st.session_state['selected_rows'].discard(idx)
    st.session_state['deselect_all_trigger'] = False
     # Clear editor state to force UI update
    if "browse_table" in st.session_state:
        del st.session_state["browse_table"]

# Prepare URL columns function (defined outside rewrite block for cleanliness)
def add_uniprot_url_columns(df, id_column, prefix):
    url_columns = []
    for idx, row in df.iterrows():
        if pd.notna(row[id_column]) and row[id_column] != '':
            ids = str(row[id_column]).split(',')
            for i, id_val in enumerate(ids):
                col_name = f'{prefix}_URL_{i+1}'
                if col_name not in df.columns:
                    df[col_name] = ""
                    url_columns.append(col_name)
                df.at[idx, col_name] = f"https://uniprot.org/uniprotkb/{id_val.strip()}/entry"
    return [col for col in df.columns if col.startswith(f'{prefix}_URL')]

# REBUILD logic
if should_rebuild_display:
    # 1. Apply selections to fresh copy based on session state
    temp_df = filtered_df.copy()
    for idx in temp_df.index:
        if idx in st.session_state['selected_rows']:
            temp_df.loc[idx, 'Selected'] = True
    
    # 2. Add URL columns and format
    display_df = temp_df
    
    # Create URL columns for E2 and E3 UniProt IDs
    e2_url_columns = add_uniprot_url_columns(display_df, 'E2_uniprot', 'E2')
    e3_url_columns = add_uniprot_url_columns(display_df, 'E3_uniprot', 'E3')

    # Remove the original UniProt columns and rename URL columns to replace them
    if 'E2_uniprot' in display_df.columns:
        display_df.drop('E2_uniprot', axis=1, inplace=True)
    if 'E3_uniprot' in display_df.columns:
        display_df.drop('E3_uniprot', axis=1, inplace=True)

    # Rename the first URL column for each protein type
    if e2_url_columns:
        display_df.rename(columns={e2_url_columns[0]: 'E2_uniprot'}, inplace=True)
        e2_url_columns[0] = 'E2_uniprot'
    if e3_url_columns:
        display_df.rename(columns={e3_url_columns[0]: 'E3_uniprot'}, inplace=True)
        e3_url_columns[0] = 'E3_uniprot'

    # Reorder columns
    cols = display_df.columns.tolist()
    uniprot_cols = []
    if 'E2_uniprot' in cols:
        cols.remove('E2_uniprot')
        uniprot_cols.append('E2_uniprot')
    if 'E3_uniprot' in cols:
        cols.remove('E3_uniprot')
        uniprot_cols.append('E3_uniprot')
    for i, col in enumerate(uniprot_cols):
        cols.insert(4 + i, col)
    
    display_df = display_df[cols]

    # Sort on initial load
    if not st.session_state.get('gene_filter') and 'gene_filter' not in st.session_state:
        display_df = display_df.sort_values(by='E2_gene', ascending=True)

    # Apply gene filter if exists
    if 'gene_filter' in st.session_state:
        filter_info = st.session_state['gene_filter']
        display_df = display_df[display_df[filter_info['column']] == filter_info['value']]

    # Store in session state
    st.session_state['cached_display_df'] = display_df
    st.session_state['last_view_signature'] = view_signature
    st.session_state['cached_e2_cols'] = e2_url_columns # Store these to config columns later
    st.session_state['cached_e3_cols'] = e3_url_columns
    
else:
    # Use cached version - this preserves the dataframe object identity
    # preventing st.data_editor from resetting scroll
    display_df = st.session_state['cached_display_df']
    e2_url_columns = st.session_state.get('cached_e2_cols', [])
    e3_url_columns = st.session_state.get('cached_e3_cols', [])

# Gene Filter UI (Display only)
if 'gene_filter' in st.session_state:
    filter_info = st.session_state['gene_filter']
    col_filter, col_clear = st.columns([4, 1])
    with col_filter:
        st.info(f"Filtered by {filter_info['column']}: **{filter_info['value']}**")
    with col_clear:
        if st.button("Clear Filter", key="clear_gene_filter"):
            del st.session_state['gene_filter']
            # Force view update
            if "last_view_signature" in st.session_state:
                del st.session_state["last_view_signature"]
            st.rerun()

def download_selected_complexes():
    """Download selected complex PDB files as a zip"""
    # Get selected complexes from session state and original df
    selected_indices = st.session_state['selected_rows']
    if not selected_indices:
        st.warning("No complexes selected. Please select at least one complex from the table.")
        return
    
    selected_complexes = filtered_df.loc[list(selected_indices), 'Complex'].tolist()
    
    # Create a progress bar and status text
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text(f"Preparing to download {len(selected_complexes)} complex(es)...")
    
    zip_buffer = io.BytesIO()
    missing_files = []
    found_files = 0
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, complex_name in enumerate(selected_complexes):
            progress = (i + 1) / len(selected_complexes)
            progress_bar.progress(progress)
            progress_text.text(f"Processing {i + 1}/{len(selected_complexes)}: {complex_name}")
            
            pdb_path = f"{util.constants.filepaths.PDB_FILES}/{complex_name}.pdb"
            if os.path.exists(pdb_path):
                with open(pdb_path, 'rb') as f:
                    zip_file.writestr(f"{complex_name}.pdb", f.read())
                found_files += 1
            else:
                missing_files.append(complex_name)
    
    progress_bar.empty()
    progress_text.empty()
    
    if missing_files:
        st.warning(f"Structure files not found for: {', '.join(missing_files)}")
        
    if found_files == 0:
        st.error("No structure files could be found for the selected complexes.")
        return
    
    zip_buffer.seek(0)
    
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

for col in e2_url_columns[1:]: 
    num = col.split('_')[-1]
    column_config[col] = st.column_config.LinkColumn(
        f"E2 UniProt_{num}",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E2 protein on UniProt"
    )

for col in e3_url_columns[1:]:
    num = col.split('_')[-1]
    column_config[col] = st.column_config.LinkColumn(
        f"E3 UniProt_{num}",
        display_text="https://uniprot.org/uniprotkb/(.*)/entry",
        help="Click to view E3 protein on UniProt"
    )


# Callback to handle selection changes properly and synchronously
def on_table_edit():
    """
    Callback for data_editor changes.
    Updates st.session_state['selected_rows'] based on user edits.
    Uses 'cached_display_df' from session state to map positional indices to real indices.
    """
    if "browse_table" not in st.session_state:
        return
        
    edited_rows = st.session_state["browse_table"].get("edited_rows", {})
    if not edited_rows:
        return
        
    # Get the dataframe that matches the editor's current view (the one we passed to it)
    data = st.session_state.get('cached_display_df')
    if data is None:
        return

    for row_idx, changes in edited_rows.items():
        if 'Selected' in changes:
            # row_idx is the integer position in the displayed dataframe
            if row_idx < len(data):
                # Get the actual index label from the dataframe
                actual_index = data.index[row_idx]
                new_val = changes['Selected']
                
                if new_val:
                    st.session_state['selected_rows'].add(actual_index)
                elif actual_index in st.session_state['selected_rows']:
                    st.session_state['selected_rows'].discard(actual_index)

# Display the dataframe using data_editor
# Determine columns to make read-only (all except Selected)
editable_cols = ['Selected']
disabled_cols = [col for col in display_df.columns if col not in editable_cols]

edited_df = st.data_editor(
    display_df,
    column_config=column_config,
    hide_index=True,
    width='stretch',
    disabled=disabled_cols,
    key="browse_table",
    on_change=on_table_edit
)

# NOTE: Direct loop to update session state removed in favor of on_change callback.

# Handle gene name filtering based on cell selection
# Access selection from session state since data_editor returns the data, not the event
selection_state = st.session_state.get("browse_table", {}).get("selection", {})




