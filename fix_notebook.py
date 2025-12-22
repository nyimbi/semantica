import nbformat

def update_phase_5(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    updated = False
    for cell in nb.cells:
        if cell.cell_type == 'code' and 'GraphReasoner' in cell.source:
            # Update instantiation to be explicit
            if 'GraphReasoner(core=core)' in cell.source:
                cell.source = cell.source.replace(
                    'GraphReasoner(core=core)',
                    'GraphReasoner(core=core, provider="groq", model="llama-3.1-8b-instant")'
                )
                updated = True
                print(f"Updated GraphReasoner instantiation in cell.")
                
    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully updated notebook: {notebook_path}")
    else:
        print("No GraphReasoner instantiation found to update.")

if __name__ == "__main__":
    notebook_path = r"c:\Users\Mohd Kaif\semantica\cookbook\use_cases\advanced_rag\01_GraphRAG_Complete.ipynb"
    update_phase_5(notebook_path)
