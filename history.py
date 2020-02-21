from toolkit import *

import pandas as pd
import numpy as np


def gpa(distr):
    grades = [4.0, 3.7, 3.3, 3.0, 2.7, 2.3, 2.0, 1.7, 1.3, 1.0, 0.7, 0]
    g_sum = 0
    for i in range(len(distr)):
        g_sum += grades[i]*distr[i]
    return g_sum/sum(distr)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datadir', default='./data/',
                    required=False, help='path to input dataset')
    ap.add_argument('-c', '--class', default='',
                    required=False, help='Department that offers the classes, default is Mathematical Sciences')
    ap.add_argument('-n', '--number', default='', required=False,
                    help='Number code of the class. (Not Section')
    ap.add_argument('-a', '--area', default='', required=False,
                    help='Area Filter, default: None (Letters & Sciences)')
    args = vars(ap.parse_args())

    # Parse datasets
    distributions = []
    gpas = []
    for i in range(4, 9):
        # Read CSV and strip column names
        df = pd.read_csv(f'{args["datadir"]}Fall_201{i}.csv').rename(
            columns=lambda x: x.strip()).fillna(0)

        # Remove Class Totals
        df = df.loc[df['Cla Class Section'] != 'Course Total']

        # Filter area
        if args['area'] != '':
            df = df.loc[df['Ag Area Sdesc'] == args['area']]

        # Filter department
        if args['class'] != '':
            df = df.loc[df['Cla Subject Ldesc'] == args['class']]

        # For reference: Avg over GPA
        gpas.append(np.average(df['gpa'].to_numpy()))

        if args['number'] != '':
            df = df.loc[df['Cla Catalog Nbr'] == int(args['number'])]

        # Sum distribution and add to array
        distributions.append(
            df[['A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'F, F+']].sum())

    matrix = build_distance_matrix(pd.Series(distributions))

    for i in range(len(distributions)):
        print(f'Number of grades given out in year 201{i+4}: {sum(distributions[i])}')
        print(f'Avg GPA computed from all grades: {gpa(distributions[i])}')
        print(f'Average of listed GPAs: {gpas[i]}')
        print()

    print('EMD Matrix multiplied with 100:')
    print(matrix*100)
