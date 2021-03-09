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

wget https://d2q79iu7y748jz.cloudfront.net/s/_squarelogo/6930c22db55ee0ebfc84ac24baffaf48 -O enhanceIT.png


echo "Create READEME.md file"
echo "Please enter your project's name:"

read projectName

echo "Please enter description for you project:"

read description

echo """# ${projectName}

${description}

""" > README.md
