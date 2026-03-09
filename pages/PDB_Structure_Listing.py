
import streamlit as st
import util.constants.filepaths
import pandas as pd
import zipfile
import io
import os
import re
from util.functions.header import header
from streamlit_js_eval import streamlit_js_eval
from util.functions.filter_dataframe import filter_dataframe
from util.functions.get_latest_pdb_table import get_latest_pdb_table
from urllib.parse import quote_plus, unquote_plus

header(__file__.split('/')[-1].split('.')[0])

st.header('Experimental Structures in PDB')

st.write('This is a collection of experimental structures from the Protein Data Bank (PDB) that include human E2 enzymes and/or E3 ligases.')

date_time, filename = get_latest_pdb_table(util.constants.filepaths.PDB_STRUCTURE_TABLE_FOLDER)

st.write(f"Updated: {date_time}")


if filename is not None:
    df = pd.read_csv(filename)
else:
    st.stop()

# Add Selected column
df.insert(0, 'Selected', False)

# Initialize session state for selections if not exists
if 'pdb_selected_rows' not in st.session_state:
    st.session_state['pdb_selected_rows'] = set()

# Filter the dataframe
filtered_df = filter_dataframe(df)

# --- Query param gene filter logic (analogous to Browse__Download) ---
params = st.query_params
if params.get("filter_col") and params.get("filter_val"):
    try:
        col = params["filter_col"]
        val = unquote_plus(params["filter_val"])
        new_filter = {"column": col, "value": val}
        st.query_params.clear()
        for k in [
            "cached_pdb_display_df",
            "pdb_last_view_signature",
            "pdb_table",
            "cached_pdb_e2_cols",
            "cached_pdb_e3_cols",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.session_state["pdb_gene_filter"] = new_filter
        st.rerun()
    except Exception:
        st.query_params.clear()

# --- View signature and triggers for select/deselect all ---
current_indices = filtered_df.index.tolist()
view_signature = f"{len(current_indices)}-{hash(tuple(current_indices))}"
bulk_action = st.session_state.get('pdb_select_all_trigger', False) or \
              st.session_state.get('pdb_deselect_all_trigger', False)
view_changed = st.session_state.get('pdb_last_view_signature') != view_signature
cache_missing = 'cached_pdb_display_df' not in st.session_state
should_rebuild_display = bulk_action or view_changed or cache_missing

# Process Select All / Deselect All logic BEFORE rebuilding
if st.session_state.get('pdb_select_all_trigger'):
    if "cached_pdb_display_df" in st.session_state:
        display_indices = st.session_state["cached_pdb_display_df"].index
    else:
        display_indices = filtered_df.index
    for idx in display_indices:
        st.session_state['pdb_selected_rows'].add(idx)
    st.session_state['pdb_select_all_trigger'] = False
    if "pdb_table" in st.session_state:
        del st.session_state["pdb_table"]

if st.session_state.get('pdb_deselect_all_trigger'):
    st.session_state['pdb_selected_rows'].clear()
    st.session_state['pdb_deselect_all_trigger'] = False
    if "pdb_table" in st.session_state:
        del st.session_state["pdb_table"]
    # Force a full page reload to ensure the data_editor clears its visual selections
    try:
        streamlit_js_eval(js_expressions="parent.window.location.reload()")
    except Exception:
        # If streamlit_js_eval is not available or fails, trigger a rerun as fallback
        if "pdb_last_view_signature" in st.session_state:
            del st.session_state["pdb_last_view_signature"]
        st.rerun()

# --- REBUILD logic ---
def add_url_columns(df, id_column, prefix):
    for idx, row in df.iterrows():
        if pd.notna(row[id_column]):
            ids = str(row[id_column]).split(',')
            for i, id_val in enumerate(ids):
                col_name = f'{prefix}_URL_{i+1}'
                if col_name not in df.columns:
                    df[col_name] = ""
                df.at[idx, col_name] = f"https://uniprot.org/uniprotkb/{id_val.strip()}/entry"
    return [col for col in df.columns if col.startswith(f'{prefix}_URL')]

if should_rebuild_display:
    temp_df = filtered_df.copy()
    for idx in temp_df.index:
        if idx in st.session_state['pdb_selected_rows']:
            temp_df.loc[idx, 'Selected'] = True
    # Apply gene filter
    if 'pdb_gene_filter' in st.session_state:
        filter_info = st.session_state['pdb_gene_filter']
        if filter_info['column'] in temp_df.columns:
            temp_df = temp_df[temp_df[filter_info['column']] == filter_info['value']]
    # Convert ubiquitin_in_structure to Yes/No on the source temp_df (simple non-blank check)
    if 'ubiquitin_in_structure' in temp_df.columns:
        temp_df['ubiquitin_in_structure'] = temp_df['ubiquitin_in_structure'].apply(
            lambda x: "Yes" if pd.notna(x) and str(x).strip() != "" else "No"
        )

    display_df = temp_df
    # Ensure PDB_URL column exists
    if 'PDB_ID' in display_df.columns and 'PDB_URL' not in display_df.columns:
        display_df['PDB_URL'] = display_df['PDB_ID'].apply(
            lambda x: f"https://www.rcsb.org/structure/{x}" if pd.notna(x) else ""
        )
    e2_url_columns = add_url_columns(display_df, 'E2_UniProtIDs', 'E2')
    e3_url_columns = add_url_columns(display_df, 'E3_UniProtIDs', 'E3')
    # Replace gene columns with query string links, as in Browse__Download
    from urllib.parse import quote_plus
    if 'E2_Gene_Names' in display_df.columns:
        display_df['__E2_gene_link__'] = display_df['E2_Gene_Names'].apply(
            lambda v: f"?filter_col=E2_Gene_Names&filter_val={quote_plus(str(v))}" if pd.notna(v) and str(v) != '' else ''
        )
        display_df.drop('E2_Gene_Names', axis=1, inplace=True)
        display_df.rename(columns={'__E2_gene_link__': 'E2_Gene_Names'}, inplace=True)
    if 'E3_Gene_Names' in display_df.columns:
        display_df['__E3_gene_link__'] = display_df['E3_Gene_Names'].apply(
            lambda v: f"?filter_col=E3_Gene_Names&filter_val={quote_plus(str(v))}" if pd.notna(v) and str(v) != '' else ''
        )
        display_df.drop('E3_Gene_Names', axis=1, inplace=True)
        display_df.rename(columns={'__E3_gene_link__': 'E3_Gene_Names'}, inplace=True)
    # Reorder columns robustly
    cols = display_df.columns.tolist()
    url_columns_to_remove = []
    if 'PDB_URL' in cols:
        url_columns_to_remove.append('PDB_URL')
    url_columns_to_remove += e2_url_columns + e3_url_columns
    id_columns_to_remove = ['PDB_ID', 'E2_UniProtIDs', 'E3_UniProtIDs', 'Ub_UniProtIDs', 'Selected']
    for col in url_columns_to_remove + id_columns_to_remove:
        if col in cols:
            cols.remove(col)
    special_columns = ['E2_Gene_Names', 'E3_Gene_Names', 'ubiquitin_in_structure']
    for col in special_columns:
        if col in cols:
            cols.remove(col)
    col_order = []
    # Ensure the selection checkbox column is the far-left column in the display
    if 'Selected' in display_df.columns:
        col_order.append('Selected')
    if 'PDB_URL' in display_df.columns:
        col_order.append('PDB_URL')
    col_order += [c for c in special_columns if c in display_df.columns]
    col_order += [c for c in e2_url_columns if c in display_df.columns]
    col_order += [c for c in e3_url_columns if c in display_df.columns]
    col_order += cols
    display_df = display_df[col_order]
    st.session_state['cached_pdb_display_df'] = display_df
    st.session_state['pdb_last_view_signature'] = view_signature
    st.session_state['cached_pdb_e2_cols'] = e2_url_columns
    st.session_state['cached_pdb_e3_cols'] = e3_url_columns
else:
    display_df = st.session_state['cached_pdb_display_df']
    e2_url_columns = st.session_state.get('cached_pdb_e2_cols', [])
    e3_url_columns = st.session_state.get('cached_pdb_e3_cols', [])


# (ubiquitin_in_structure is converted earlier on temp_df)



# --- Column config ---
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
    ),
    "Selected": st.column_config.CheckboxColumn()
}




# Use LinkColumn with regex to display only the gene name from the query string, as in Browse__Download
column_config['E2_Gene_Names'] = st.column_config.LinkColumn(
    "E2_Gene_Names",
    display_text=r".*filter_val=([^&]+).*",
    help="Click to filter by this E2 gene"
)
column_config['E3_Gene_Names'] = st.column_config.LinkColumn(
    "E3_Gene_Names",
    display_text=r".*filter_val=([^&]+).*",
    help="Click to filter by this E3 gene"
)


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
editable_cols = ['Selected']
disabled_cols = [col for col in display_df.columns if col not in editable_cols]

for col in display_df.columns:
    if col not in column_config:
        column_config[col] = st.column_config.Column(
            col,
            help=f"{col} information",
            width="auto"
        )



# --- Download logic (with progress and missing file handling) ---


# --- Download logic with user choice for file type (two-step process) ---
def download_selected_pdbs():
    selected_indices = st.session_state['pdb_selected_rows']
    if not selected_indices:
        st.warning("No structures selected. Please select at least one structure from the table.")
        return
    st.session_state['show_download_options'] = True

def perform_download(index_list, file_choice, file_source="rcsb"):
    # index_list: list of dataframe index labels corresponding to filtered_df
    progress_text = st.empty()
    progress_bar = st.progress(0)
    progress_text.text(f"Preparing to download {len(index_list)} structure(s)...")
    zip_buffer = io.BytesIO()
    missing_files = []
    found_files = 0
    # Select cache directory based on source
    if file_source == "pdbrenum":
        pdb_cache_dir = util.constants.filepaths.PDBRENUM_CIF_FILES
    else:
        pdb_cache_dir = util.constants.filepaths.PDB_CIF_FILES
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i, idx in enumerate(index_list):
            # Safely get PDB_ID and gene info from filtered_df
            try:
                pdbid = str(filtered_df.loc[idx, 'PDB_ID'])
            except Exception:
                pdbid = ''
            try:
                e2_raw = filtered_df.loc[idx, 'E2_Gene_Names'] if 'E2_Gene_Names' in filtered_df.columns else ''
            except Exception:
                e2_raw = ''
            try:
                e3_raw = filtered_df.loc[idx, 'E3_Gene_Names'] if 'E3_Gene_Names' in filtered_df.columns else ''
            except Exception:
                e3_raw = ''
            try:
                ub_flag = filtered_df.loc[idx, 'ubiquitin_in_structure'] if 'ubiquitin_in_structure' in filtered_df.columns else ''
            except Exception:
                ub_flag = ''

            # Helper to sanitize gene names for filenames
            def _sanitize(name):
                if not pd.notna(name):
                    return []
                s = str(name)
                # normalize separators and split
                s = s.replace(';', ',')
                parts = [p.strip() for p in s.split(',') if p.strip()]
                safe_parts = []
                for p in parts:
                    sp = re.sub(r"[^A-Za-z0-9_\-]+", "_", p)
                    sp = sp.strip('_')
                    if sp:
                        safe_parts.append(sp)
                return safe_parts

            e2_names = _sanitize(e2_raw)
            e3_names = _sanitize(e3_raw)
            name_parts = []
            ub_present = pd.notna(ub_flag) and str(ub_flag).strip() != ''
            if ub_present:
                name_parts.append('Ub')
            name_parts += e2_names
            name_parts += e3_names
            name_parts.append(pdbid)
            # final base filename
            base_fname = '_'.join([p for p in name_parts if p])
            progress = (i + 1) / len(index_list)
            progress_bar.progress(progress)
            progress_text.text(f"Processing {i + 1}/{len(index_list)}: {pdbid}")
            files_to_add = []
            if file_choice in ("asym", "both"):
                asym_path = os.path.join(pdb_cache_dir, f"{pdbid}.cif")
                if pdbid and os.path.exists(asym_path):
                    arcname = f"{base_fname}_asym.cif"
                    files_to_add.append((asym_path, arcname))
                else:
                    missing_files.append(f"{pdbid or '<missing id>'} (asym)")
            if file_choice in ("bio", "both"):
                bio_path = os.path.join(pdb_cache_dir, f"{pdbid}-assembly1.cif")
                if pdbid and os.path.exists(bio_path):
                    arcname = f"{base_fname}_bioassembly.cif"
                    files_to_add.append((bio_path, arcname))
                else:
                    missing_files.append(f"{pdbid or '<missing id>'} (bioassembly)")
            for src, arcname in files_to_add:
                try:
                    with open(src, 'rb') as f:
                        zip_file.writestr(arcname, f.read())
                        found_files += 1
                except Exception:
                    missing_files.append(f"{pdbid or '<missing id>'} (read error)")
    progress_bar.empty()
    progress_text.empty()
    if missing_files:
        st.warning(f"Some files not found: {', '.join(missing_files)}")
    if found_files == 0:
        st.error("No structure files could be found for the selected structures.")
        return
    zip_buffer.seek(0)
    st.download_button(
        label=f"Download {found_files} CIF Files",
        data=zip_buffer,
        file_name="selected_structures.zip",
        mime="application/zip",
        key="download_cif_files_pdbpage"
    )
    st.success(f"{found_files} structure CIF files prepared for download.")


# --- UI Buttons ---

col1, col2, col3, col4 = st.columns([2, 0.8, 0.9, 4.8])
with col1:
    if st.button("Download Selected Structures"):
        download_selected_pdbs()
with col2:
    if st.button("Select All", key="pdb_select_all_btn"):
        st.session_state['pdb_select_all_trigger'] = True
        st.rerun()
with col3:
    if st.button("Deselect All", key="pdb_deselect_all_btn"):
        st.session_state['pdb_deselect_all_trigger'] = True
        st.rerun()

# --- Download options modal ---
if st.session_state.get('show_download_options', False):
    selected_indices = st.session_state['pdb_selected_rows']
    selected_rows = display_df.loc[list(selected_indices)]
    st.info("Choose file source and format for the selected structures:")
    
    col1, col2 = st.columns(2)
    with col1:
        file_source = st.radio(
            "PDB File Source",
            options=["rcsb", "pdbrenum"],
            format_func=lambda x: "RCSB PDB (standard)" if x == "rcsb" else "PDBrenum (renumbered)",
            key="pdb_file_source_radio"
        )
    with col2:
        file_type = st.radio(
            "File Format",
            ["Asymmetric unit only (.cif)", "Biological assemblies only (-assembly1.cif)", "Both"],
            key="download_file_type_radio"
        )
    
    if file_type == "Asymmetric unit only (.cif)":
        file_choice = "asym"
    elif file_type == "Biological assemblies only (-assembly1.cif)":
        file_choice = "bio"
    else:
        file_choice = "both"
    
    if st.button("Confirm and Download", key="confirm_download_btn"):
        st.session_state['show_download_options'] = False
        perform_download(list(selected_indices), file_choice, file_source=file_source)
    if st.button("Cancel", key="cancel_download_btn"):
        st.session_state['show_download_options'] = False

# --- Gene Filter UI ---
if 'pdb_gene_filter' in st.session_state:
    filter_info = st.session_state['pdb_gene_filter']
    col_filter, col_clear = st.columns([4, 1])
    with col_filter:
        st.info(f"Filtered by {filter_info['column']}: **{filter_info['value']}**")
    with col_clear:
        if st.button("Clear Filter", key="clear_pdb_gene_filter"):
            del st.session_state['pdb_gene_filter']
            if "pdb_last_view_signature" in st.session_state:
                del st.session_state["pdb_last_view_signature"]
            st.rerun()

# --- Data Editor with selection ---
def on_pdb_table_edit():
    if "pdb_table" not in st.session_state:
        return
    edited_rows = st.session_state["pdb_table"].get("edited_rows", {})
    if not edited_rows:
        return
    data = st.session_state.get('cached_pdb_display_df')
    if data is None:
        return
    for row_idx, changes in edited_rows.items():
        if 'Selected' in changes:
            if row_idx < len(data):
                actual_index = data.index[row_idx]
                new_val = changes['Selected']
                if new_val:
                    st.session_state['pdb_selected_rows'].add(actual_index)
                elif actual_index in st.session_state['pdb_selected_rows']:
                    st.session_state['pdb_selected_rows'].discard(actual_index)

edited_df = st.data_editor(
    display_df,
    disabled=disabled_cols,
    hide_index=True,
    column_config=column_config,
    use_container_width=True,
    key="pdb_table",
    on_change=on_pdb_table_edit
)

# Sync selections from the returned editor dataframe to session state to avoid intermittent unchecking
try:
    if edited_df is not None and 'Selected' in edited_df.columns:
        # edited_df has same ordering as display_df passed in; map positional rows to actual indices
        for pos in range(len(edited_df)):
            actual_index = edited_df.index[pos]
            val = bool(edited_df.iloc[pos]['Selected'])
            if val:
                st.session_state['pdb_selected_rows'].add(actual_index)
            elif actual_index in st.session_state['pdb_selected_rows']:
                st.session_state['pdb_selected_rows'].discard(actual_index)
except Exception:
    pass

# Simple row-selection-based filtering UI (match Browse__Download behavior)
pdb_state = st.session_state.get("pdb_table", {})
selected_positions = pdb_state.get("selected_rows", []) or []
if selected_positions and len(selected_positions) == 1:
    pos = selected_positions[0]
    if isinstance(pos, int) and 0 <= pos < len(display_df):
        row = display_df.iloc[pos]
        c1, c2 = st.columns([1, 1])
        with c1:
            if 'E2_Gene_Names' in display_df.columns and st.button(f"Filter by E2: {row['E2_Gene_Names']}", key=f"filter_e2_{pos}"):
                st.session_state['pdb_gene_filter'] = {'column': 'E2_Gene_Names', 'value': row['E2_Gene_Names']}
                if "pdb_last_view_signature" in st.session_state:
                    del st.session_state["pdb_last_view_signature"]
                try:
                    st.session_state["pdb_table"]["selected_rows"] = []
                except Exception:
                    pass
                st.rerun()
        with c2:
            if 'E3_Gene_Names' in display_df.columns and st.button(f"Filter by E3: {row['E3_Gene_Names']}", key=f"filter_e3_{pos}"):
                st.session_state['pdb_gene_filter'] = {'column': 'E3_Gene_Names', 'value': row['E3_Gene_Names']}
                if "pdb_last_view_signature" in st.session_state:
                    del st.session_state["pdb_last_view_signature"]
                try:
                    st.session_state["pdb_table"]["selected_rows"] = []
                except Exception:
                    pass
                st.rerun()
