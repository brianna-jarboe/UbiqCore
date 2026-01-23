import streamlit as st
import util.constants.filepaths
import pandas as pd
import zipfile
import io
import os
import hashlib
from util.functions.header import header
from util.functions.filter_dataframe import filter_dataframe
from urllib.parse import quote_plus, unquote_plus
from streamlit_js_eval import streamlit_js_eval

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

# --- apply query params early so clicks that set query params rebuild/sort correctly ---
params = st.query_params
if params.get("filter_col") and params.get("filter_val"):
    try:
        col = params["filter_col"]
        val = unquote_plus(params["filter_val"])
        new_filter = {"column": col, "value": val}
        # Always clear query params immediately to avoid repeated filter on rerun
        st.query_params.clear()
        # Remove all relevant session state to force a full rebuild
        for k in [
            "cached_display_df",
            "last_view_signature",
            "browse_table",
            "cached_e2_cols",
            "cached_e3_cols",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["gene_filter"] = new_filter
        st.rerun()
    except Exception:
        st.query_params.clear()
# --- end new ---

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
    # Select all rows currently displayed in the table (after all filtering and sorting)
    if "cached_display_df" in st.session_state:
        display_indices = st.session_state["cached_display_df"].index
    else:
        display_indices = filtered_df.index
    for idx in display_indices:
        st.session_state['selected_rows'].add(idx)
    st.session_state['select_all_trigger'] = False
    if "browse_table" in st.session_state:
        del st.session_state["browse_table"]

if st.session_state.get('deselect_all_trigger'):
    # Deselect ALL rows (not just displayed) by clearing the set
    st.session_state['selected_rows'].clear()
    st.session_state['deselect_all_trigger'] = False
    if "browse_table" in st.session_state:
        del st.session_state["browse_table"]
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

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

    # --- Apply gene filter EARLY on the original columns (so equality checks use raw values) ---
    if 'gene_filter' in st.session_state:
        filter_info = st.session_state['gene_filter']
        if filter_info['column'] in temp_df.columns:
            temp_df = temp_df[temp_df[filter_info['column']] == filter_info['value']]

    # 2. Prepare display dataframe (keep raw gene columns for filtering/sorting)
    display_df = temp_df

    # Create UniProt URL columns and rename primary ones
    e2_url_columns = add_uniprot_url_columns(display_df, 'E2_uniprot', 'E2')
    e3_url_columns = add_uniprot_url_columns(display_df, 'E3_uniprot', 'E3')

    # If raw "E2_uniprot"/"E3_uniprot" are present and we generated URL columns, drop the raw to avoid name collisions,
    # then rename the first URL column to the friendly name.
    if e2_url_columns and 'E2_uniprot' in display_df.columns:
        display_df.drop('E2_uniprot', axis=1, inplace=True)
    if e3_url_columns and 'E3_uniprot' in display_df.columns:
        display_df.drop('E3_uniprot', axis=1, inplace=True)

    if e2_url_columns:
        display_df.rename(columns={e2_url_columns[0]: 'E2_uniprot'}, inplace=True)
    if e3_url_columns:
        display_df.rename(columns={e3_url_columns[0]: 'E3_uniprot'}, inplace=True)

    # Create clickable gene link values for display (keep raw gene values in temp_df for filter/sort)
    if 'E2_gene' in display_df.columns:
        display_df['__E2_gene_link__'] = display_df['E2_gene'].apply(
            lambda v: f"?filter_col=E2_gene&filter_val={quote_plus(str(v))}" if pd.notna(v) and str(v) != "" else ""
        )
        # drop raw display column and replace with link-named column
        display_df.drop('E2_gene', axis=1, inplace=True)
        display_df.rename(columns={'__E2_gene_link__': 'E2_gene'}, inplace=True)

    if 'E3_gene' in display_df.columns:
        display_df['__E3_gene_link__'] = display_df['E3_gene'].apply(
            lambda v: f"?filter_col=E3_gene&filter_val={quote_plus(str(v))}" if pd.notna(v) and str(v) != "" else ""
        )
        display_df.drop('E3_gene', axis=1, inplace=True)
        display_df.rename(columns={'__E3_gene_link__': 'E3_gene'}, inplace=True)

    # Recompute the actual URL column lists (now includes renamed primary column)
    e2_url_columns = [c for c in display_df.columns if c.startswith('E2_uniprot') or c.startswith('E2_URL')]
    e3_url_columns = [c for c in display_df.columns if c.startswith('E3_uniprot') or c.startswith('E3_URL')]

    # Ensure column names are unique (data_editor requires unique names).
    def _ensure_unique_cols(df):
        cols = df.columns.tolist()
        if len(set(cols)) == len(cols):
            return df
        seen = {}
        new_cols = []
        for c in cols:
            if c in seen:
                seen[c] += 1
                new_c = f"{c}_{seen[c]}"
                while new_c in seen:
                    seen[c] += 1
                    new_c = f"{c}_{seen[c]}"
                new_cols.append(new_c)
                seen[new_c] = 1
            else:
                new_cols.append(c)
                seen[c] = 1
        df.columns = new_cols
        return df

    display_df = _ensure_unique_cols(display_df)

    # Reorder so gene + its UniProt columns sit right after the first Complex column
    cols = display_df.columns.tolist()
    desired_after_complex = []
    if 'E2_gene' in display_df.columns:
        desired_after_complex.extend(['E2_gene'] + [c for c in e2_url_columns if c in display_df.columns])
    if 'E3_gene' in display_df.columns:
        desired_after_complex.extend(['E3_gene'] + [c for c in e3_url_columns if c in display_df.columns])
    if 'Complex' in cols and desired_after_complex:
        insert_idx = cols.index('Complex') + 1
        for c in desired_after_complex:
            if c in cols:
                cols.remove(c)
        for c in desired_after_complex:
            if c in display_df.columns:
                cols.insert(insert_idx, c)
                insert_idx += 1
    display_df = display_df[cols]

    # Sort on initial load (only when no gene_filter) using raw gene values from temp_df
    if not st.session_state.get('gene_filter') and 'gene_filter' not in st.session_state:
        if 'E2_gene' in temp_df.columns:
            display_df = display_df.loc[temp_df.sort_values(by='E2_gene', ascending=True).index]
        else:
            display_df = display_df

    # Store in session state
    st.session_state['cached_display_df'] = display_df
    st.session_state['last_view_signature'] = view_signature
    st.session_state['cached_e2_cols'] = e2_url_columns
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

# Register gene columns as LinkColumn (clicks set query params handled earlier)
if 'E2_gene' in display_df.columns:
	column_config['E2_gene'] = st.column_config.LinkColumn(
		"E2_gene",
		display_text=r".*filter_val=([^&]+).*",
		help="Click to filter by this E2 gene"
	)
if 'E3_gene' in display_df.columns:
	column_config['E3_gene'] = st.column_config.LinkColumn(
		"E3_gene",
		display_text=r".*filter_val=([^&]+).*",
		help="Click to filter by this E3 gene"
	)

# Register UniProt link columns
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
    on_change=on_table_edit,  # removed unsupported selection_mode arg
)

# NOTE: Direct loop to update session state removed in favor of on_change callback.

# Simple row-selection-based filtering UI:
browse_state = st.session_state.get("browse_table", {})
selected_positions = browse_state.get("selected_rows", []) or []
if selected_positions and len(selected_positions) == 1:
    pos = selected_positions[0]
    if isinstance(pos, int) and 0 <= pos < len(display_df):
        row = display_df.iloc[pos]
        c1, c2 = st.columns([1, 1])
        with c1:
            if 'E2_gene' in display_df.columns and st.button(f"Filter by E2: {row['E2_gene']}", key=f"filter_e2_{pos}"):
                st.session_state['gene_filter'] = {'column': 'E2_gene', 'value': row['E2_gene']}
                if "last_view_signature" in st.session_state:
                    del st.session_state["last_view_signature"]
                # clear selection to avoid repeated immediate UI
                try:
                    st.session_state["browse_table"]["selected_rows"] = []
                except Exception:
                    pass
                st.rerun()
        with c2:
            if 'E3_gene' in display_df.columns and st.button(f"Filter by E3: {row['E3_gene']}", key=f"filter_e3_{pos}"):
                st.session_state['gene_filter'] = {'column': 'E3_gene', 'value': row['E3_gene']}
                if "last_view_signature" in st.session_state:
                    del st.session_state["last_view_signature"]
                try:
                    st.session_state["browse_table"]["selected_rows"] = []
                except Exception:
                    pass
                st.rerun()




