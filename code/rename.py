import os

# Define the base path to your dataset
base_path = "datasat orginal"

# Mapping of abbreviations to full names
abbreviation_map = {
    'BA': 'Basophil',
    'BNE': 'Banded Neutrophil',
    'EO': 'Eosinophil',
    'ERB': 'Erythroblast',
    'LY': 'Lymphocyte',
    'MMY': 'Meta-myelocyte',
    'MO': 'Monocyte',
    'MY': 'Myelocyte',
    'PLT': 'Platelet',
    'PMY': 'Pro-myelocyte',
    'SNE': 'Segmented Neutrophil'
}

# Rename folders in Train, Test, and Validate directories
for split in ['Train', 'Test', 'Validate']:
    split_path = os.path.join(base_path, split)
    
    if os.path.exists(split_path):
        for abbrev, full_name in abbreviation_map.items():
            old_path = os.path.join(split_path, abbrev)
            new_path = os.path.join(split_path, full_name)
            
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f"Renamed {split}/{abbrev} to {split}/{full_name}")
            else:
                print(f"Folder {split}/{abbrev} does not exist")

print("Renaming completed!")