"""
Cybersecurity Intrusion Detection Optimization
Project Structure Setup Script

This creates the necessary directory structure for the project.
"""

import os

def create_project_structure():
    """Create the project directory structure"""
    
    directories = [
        'data',
        'src',
        'notebooks',
        'reports/figures',
        'reports/results',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files
    init_files = [
        'src/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization\n')
        print(f"Created file: {init_file}")
    
    print("\nProject structure created successfully!")

if __name__ == "__main__":
    create_project_structure()