import py3Dmol
from stmol import showmol
import streamlit as st
import os

def show_3D_struc(pdb_file, height=500, width=None, bgcolor='#FEE1E6'):
    """
    Display a 3D structure from a PDB file.
    
    Parameters:    - pdb_file: Path to the PDB file
    - height: Height of the viewer in pixels (default: 500)
    - width: Width of the viewer in pixels (if None, default to 700px)
    - bgcolor: Background color of the viewer (default: '#FEE1E6')
    """
    # Extract chain names from filename
    filename = os.path.basename(pdb_file)
    # Remove .pdb extension and split by hyphens
    chain_names = filename.replace('.pdb', '').split('-')
      # Ensure we have at least 3 chain names, pad with default names if needed
    while len(chain_names) < 3:
        chain_names.append(f"Chain{len(chain_names) + 1}")
    
    # PDB visualization
    view = py3Dmol.view()
    view.addModel(open(pdb_file, 'r').read(), 'pdb')
    
    for n, chain, color, chain_name in zip(range(3), list("ABC"),
                              ["#FAADA8", "#D1A2DF", "#A38DE5"], chain_names):
        view.setStyle({'chain': chain}, {'cartoon': {'color': color}})
      # Add labels after styling all chains
    for i, (chain, chain_name) in enumerate(zip(list("ABC"), chain_names)):
        # Position labels to the side with offsets to avoid blocking the structure
        # Different positions for each chain
        if i == 0:  # Chain A - position to upper left
            position = {'x': -20, 'y': 15, 'z': 0}
        elif i == 1:  # Chain B - position to upper right  
            position = {'x': 20, 'y': 15, 'z': 0}
        else:  # Chain C - position to lower center
            position = {'x': 0, 'y': -20, 'z': 0}
            
        view.addLabel(chain_name, 
                     {'fontColor': 'black', 'fontSize': 16, 'fontOpacity': 1,
                      'backgroundColor': 'white', 'backgroundOpacity': 0.65,
                      'borderColor': 'black', 'borderOpacity': 1,
                      'position': position},
                     {'chain': chain})
    
    view.addSurface(py3Dmol.VDW, {"opacity": 0.5, "color": "#A38DE5"}, {"hetflag": False})
    view.zoomTo()
    view.setBackgroundColor(bgcolor)
    
    # Set default width if not provided or if it's a function
    if width is None:
        # Use a reasonable default width for container
        width = 700
    elif callable(width):
        # Fallback if width is a function
        width = 700
    elif isinstance(width, str):
        # Convert string values to numeric
        try:
            # If it's something like '700px', extract the number
            width = int(''.join(filter(str.isdigit, width)))
        except ValueError:
            # Default if conversion fails
            width = 700
    
    # Ensure height is numeric
    if callable(height):
        height = 500
    elif isinstance(height, str):
        try:
            height = int(''.join(filter(str.isdigit, height)))
        except ValueError:
            height = 500
        
    # Display the molecule
    showmol(view, height=500, width=800)