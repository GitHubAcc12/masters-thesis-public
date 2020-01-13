# Prerequisites 
- Python 3.6 or higher
- most recent versions of packages:
    - numpy
    - pandas
    - networkx
    - matplotlib
    - argparse

Just run `pip install -r requirements.txt` to ensure they are all installed.

# Execute
Call `python toolkit.py -d "path to datafile" -t "threshold for connected components distance"` from a command shell. Depending on which version of python the environment variable points to, you might need to call python3 instead of python.
The arguments `-d` and `-t` are optional, the default threshold is 0.02 and the default dataset is the one included in the repository.