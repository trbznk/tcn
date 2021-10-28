import shutil

source = "./"
destination = "/Users/alex/gdrive_alex/Projekte/tsf/"
shutil.copytree(source, destination, dirs_exist_ok=True)
shutil.rmtree(f"{destination}src/__pycache__")
