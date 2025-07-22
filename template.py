import os

files_list = [
    "config.py",
    ".env",
    "README.md",
    "requirements.txt",
    "main.py",
    "app.py",
    "setup.py",
    ".gitignore",
    "data/base.txt",
    "templates/base.html",
    "QASystem/__init__.py",
    "QASystem/ingestion.py",
    "QASystem/retrivalAndAugumented.py",
    "QASystem/utils.py"
]

for file in files_list:
    file_path = os.path.join(os.path.dirname(__file__), file)
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write("")
    