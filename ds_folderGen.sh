#! /bin/bash

echo "Create Data Science Structure"
echo "Create data folder"
mkdir data
touch data/.gitkeep
echo "Create report folder"
mkdir report
echo "Create notebook folder"
mkdir notebook
echo "Create src folder"
mkdir src
echo "Create config.py"
touch config.py

echo """from pathlib import Path

file_dir = Path(__file__).resolve().parent

class CONFIG:
    data = file_dir.parent / 'data'
    report = file_dir.parent / 'report'
    notebook = file_dir.parent / 'notebook'
    src = file_dir.parent / 'src'

if __name__ == '__main__':
    print(file_dir)
""" > notebook/config.py

echo "Create .gitignore"
echo "data/*\n" > .gitignore

LOGO = "C:/Users/tnguy/OneDrive/Desktop/EnhanceIT_expenses/enhanceITLogo.png"

cp $LOGO ./data