from toolkit import *

import pandas as pd
import numpy as np


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datadir', default='./data/',
                    required=False, help='path to input dataset')
    ap.add_argument('-c', '--class', default='Mathematical Sciences',
                    required=False, help='Classes from which department?')
    ap.add_argument('-n', '--number', default='', required=False,
                    help='Look at specific classes from the department. Default department is Math.')
    args = vars(ap.parse_args())

    # Parse datasets
    distributions = []
    for i in range(4, 9):
        # Read CSV and strip column names
        df = pd.read_csv(f'{args["datadir"]}Fall_201{i}.csv').rename(
            columns=lambda x: x.strip())

        # Remove Class totals and filter department
        df = df.loc[(df['Cla Class Section'] != 'Course Total') &
                    (df['Cla Subject Ldesc'] == args['class'])]

        if args['number'] != '':
            df = df.loc[df['Cla Catalog Nbr'] == int(args['number'])]

        # Sum distribution and add to array
        distributions.append(
            df[['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F, F+']].sum())

    matrix = build_distance_matrix(pd.Series(distributions))
    print(matrix)
