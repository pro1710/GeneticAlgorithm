import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import sys



if __name__ == '__main__':
    # filenames = input("Enter first filenames:")
    filenames = 'logs/log_24400_2019-10-27-13.55.59.txt '
    first_filename, second_filename = tuple(filenames.split())

    first_df = pd.read_csv(first_filename)
    second_df = pd.read_csv(second_filename)

    plt.plot(first_df['GENERATION'], first_df['SCORE'])
    plt.plot(second_df['GENERATION'], second_df['SCORE'])
    plt.show()


