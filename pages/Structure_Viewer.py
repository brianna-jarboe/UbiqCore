import streamlit as st
import util.constants.filepaths
import pandas as pd
import os
from streamlit_js_eval import streamlit_js_eval
from util.functions.header import header
from util.functions.struc_3D import show_3D_struc
from util.functions.filter_dataframe import filter_dataframe

header(__file__.split('/')[-1].split('.')[0])

def structure_viewer(pdb_file, width=None):
    # Check if the file exists before trying to show it
    if os.path.exists(pdb_file):
        # Wrap the 3D structure in a centered container
        with st.container():
            st.markdown('<div class="centered-container">', unsafe_allow_html=True)
            # Use larger dimensions to maximize the viewing area
            show_3D_struc(pdb_file, bgcolor="#FDD1D7", width=1600, height=1000)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"No structure for complex: {os.path.basename(pdb_file).replace('.pdb', '')} available.")

def parse_complex_name(complex_name):
    """Parse E2 and E3 from complex name"""
    parts = complex_name.split("-")
    
    # Make sure we have enough parts after splitting
    if len(parts) >= 3:
        e2_name = parts[1]
        e3_name = parts[2]
    else:
        e2_name = "Error: Could not extract E2"
        e3_name = "Error: Could not extract E3"
    
    return e2_name, e3_name

def on_complex_change(viewer_id):
    """Callback when complex name changes to update E2 and E3"""
    # Get complex name and convert to uppercase
    complex_name = st.session_state[f"complex_{viewer_id}"].upper()
    
    # Store the uppercase version back in session state
    st.session_state[f"complex_{viewer_id}"] = complex_name
    
    # Also update the selected_viewer variable to maintain consistency
    # between manual input and dataframe selection
    st.session_state[f"selected_{viewer_id}"] = complex_name
    
    e2_name, e3_name = parse_complex_name(complex_name)
    
    # Update E2 and E3 in session state
    st.session_state[f"e2_{viewer_id}"] = e2_name
    st.session_state[f"e3_{viewer_id}"] = e3_name
    
    # The structure will be updated on rerun
    
def on_e2_e3_change(viewer_id):
    """Callback when E2 or E3 is changed to update the complex name"""
    # Get the current values and convert to uppercase
    e2_name = st.session_state[f"e2_{viewer_id}"].upper()
    e3_name = st.session_state[f"e3_{viewer_id}"].upper()
    
    # Store the uppercase versions back in session state
    st.session_state[f"e2_{viewer_id}"] = e2_name
    st.session_state[f"e3_{viewer_id}"] = e3_name
    
    # Update the complex name - assume UB is the first part
    complex_name = f"UB-{e2_name}-{e3_name}"
    st.session_state[f"complex_{viewer_id}"] = complex_name
    
    # Also update the selected_viewer variable to maintain consistency
    # between manual input and dataframe selection
    st.session_state[f"selected_{viewer_id}"] = complex_name

def create_structure_viewer(col, viewer_id, initial_complex=None):
    """Create a structure viewer with E2 and E3 text inputs that update when complex name changes"""
    # Check if there's a selected structure from the dataframe
    if f"selected_{viewer_id}" in st.session_state:
        initial_complex = st.session_state[f"selected_{viewer_id}"]
        # Always update the complex name with the selection
        st.session_state[f"complex_{viewer_id}"] = initial_complex
        
    # Initialize in session state if not already there
    if f"complex_{viewer_id}" not in st.session_state:
        st.session_state[f"complex_{viewer_id}"] = initial_complex or "UB-UBE2D2-TRIM5"
    
    # Get current complex from session state
    complex_name = st.session_state[f"complex_{viewer_id}"]
      # Extract E2 and E3 from complex name
    e2_name, e3_name = parse_complex_name(complex_name)
    
    # Store in session state for other components to access
    st.session_state[f"e2_{viewer_id}"] = e2_name
    st.session_state[f"e3_{viewer_id}"] = e3_name
    
    with col.container(border=True):
        # Complex input with on_change callback
        st.text_input(" Complex", 
                     key=f"complex_{viewer_id}",
                     on_change=on_complex_change,
                     args=(viewer_id,))
        
        # E2 and E3 input fields - now editable
        e2_col, e3_col = st.columns([1, 1])
        with e2_col:
            st.text_input("   E2", 
                         key=f"e2_{viewer_id}",
                         on_change=on_e2_e3_change,
                         args=(viewer_id,),
                         help="Edit this field to change E2 and update the structure")
        with e3_col:
            st.text_input("   E3", 
                         key=f"e3_{viewer_id}",
                         on_change=on_e2_e3_change,
                         args=(viewer_id,),
                         help="Edit this field to change E3 and update the structure")
  
        # Use the current complex name to show structure
        structure_viewer(f"{util.constants.filepaths.PDB_FILES}/{complex_name}.pdb")


st.header('Structure Viewer')




st.write('Enter the names of the E2(s) and E3(s) of interest, or select complexes from the filterable table below, to view and compare structures')

col1, col2 = st.columns([1, 1])

# Check if we have selections from the dataframe
if "selected_viewer1" in st.session_state:
    initial_complex1 = st.session_state.selected_viewer1
else:
    initial_complex1 = "UB-UBE2D2-TRIM5"
    
if "selected_viewer2" in st.session_state:
    initial_complex2 = st.session_state.selected_viewer2
else:
    initial_complex2 = "UB-UBE2N-CBL"

# Not using col_width anymore as we're calculating from screen width
# Create two structure viewers with unique IDs and initial complexes
create_structure_viewer(col1, "viewer1", initial_complex1)  
create_structure_viewer(col2, "viewer2", initial_complex2)





def handle_structure_selection(edited_df):
    """
    Handle selection of structures from the dataframe.
    When users select rows:
    - First selection goes to viewer1
    - Second selection goes to viewer2
    - Third selection replaces first selection
    """
    # Check if the 'Selected' column exists in edited_df
    if 'Selected' not in edited_df.columns:
        return
        
    # Get the current dataframe from session state
    if "structure_df" in st.session_state:
        df = st.session_state.structure_df
        
        # Make sure the 'Selected' column exists in previous df
        if 'Selected' not in df.columns:
            # If not present, just use an empty dataframe for comparison
            prev_selected = pd.DataFrame()
        else:
            # Get previously selected rows
            prev_selected = df[df['Selected'] == True]
        
        # Get newly selected rows
        curr_selected = edited_df[edited_df['Selected'] == True]
        
        # If nothing changed, return
        if len(prev_selected) == len(curr_selected) and all(prev_selected.index.isin(curr_selected.index)):
            return
            
        # If more than 2 rows are selected, keep only the two most recent
        if len(curr_selected) > 2:
            # Find the new selection by comparing with previous
            new_idx = set(curr_selected.index) - set(prev_selected.index)
            if new_idx:
                new_idx = list(new_idx)[0]  # Get the index of new selection
                
                # If already have 2 selected, uncheck the oldest one
                if len(prev_selected) == 2:
                    # Determine which one to uncheck (the one not in viewer2)
                    viewer2_complex = st.session_state.get("selected_viewer2", "")
                    
                    # Look for row that matches viewer2 complex to keep it checked
                    for idx, row in prev_selected.iterrows():
                        complex_name = row['Complex']
                        if complex_name != viewer2_complex:
                            # This is the one to uncheck
                            edited_df.at[idx, 'Selected'] = False
                
            # Make sure only 2 are selected
            selected_indices = edited_df[edited_df['Selected'] == True].index
            if len(selected_indices) > 2:
                # Keep only the 2 most recently selected
                to_keep = selected_indices[-2:]
                for idx in selected_indices:
                    if idx not in to_keep:
                        edited_df.at[idx, 'Selected'] = False
        
        # Store selected complexes in session state for next rerun
        selected_rows = edited_df[edited_df['Selected'] == True]
        selection_changed = False
        
        if len(selected_rows) >= 1:
            # Store first selection for viewer1
            complex1 = selected_rows.iloc[0]['Complex']
            # Check if this is a new selection
            if st.session_state.get("selected_viewer1", "") != complex1:
                selection_changed = True
                st.session_state["selected_viewer1"] = complex1
            
        if len(selected_rows) >= 2:
            # Store second selection for viewer2
            complex2 = selected_rows.iloc[1]['Complex']
            # Check if this is a new selection
            if st.session_state.get("selected_viewer2", "") != complex2:
                selection_changed = True
                st.session_state["selected_viewer2"] = complex2
        
        # If selections changed, force a rerun after updating session state
        if selection_changed:
            st.rerun()
    
    # Update the dataframe in session state
    st.session_state.structure_df = edited_df

# Filterable Dataframe
df = pd.read_csv(util.constants.filepaths.MAIN_DATASET_FILE)
# Add a column of 'false' to the dataframe in the beginning
df.insert(0, 'Selected', False)
# sort column by 'Complex'
df.sort_values(by='Complex', inplace=True)

# Store df in session state for comparison
if "structure_df" not in st.session_state:
    st.session_state.structure_df = df.copy()

edited_df = st.data_editor(
    filter_dataframe(df),
    column_config={
        "Selected": st.column_config.CheckboxColumn(
        )
    },
    hide_index=True,
    key="structure_editor",
    #disable all but 'Selected', should be a list of column names like ["Complex", "OtherColumn", "AnotherColumn"]
    disabled=list(df.columns.drop('Selected')),
    width='stretch'
)

# Also call the handler function for initial setup
handle_structure_selection(edited_df)

# Force rerun to apply changes when selections have changed
if "structure_df" in st.session_state:
    prev_df = st.session_state.structure_df
    
    # Make sure both dataframes have the 'Selected' column
    if 'Selected' in edited_df.columns and 'Selected' in prev_df.columns:
        currently_selected = edited_df[edited_df["Selected"] == True]
        previously_selected = prev_df[prev_df["Selected"] == True]
        
        # Check if selection has changed
        selection_changed = False
        
        if len(currently_selected) != len(previously_selected):
            selection_changed = True
        elif len(currently_selected) > 0 and len(previously_selected) > 0:
            # Check if the indices match (careful comparison to avoid errors)
            if not all(idx in previously_selected.index for idx in currently_selected.index):
                selection_changed = True
        
        if selection_changed:
            st.rerun()
