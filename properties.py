import numpy as np

# Number of users in the anonymized dataset
NUM_OF_USERS = 100

# Rate of background knowledge to anonymized dataset
ADVERSARY_RATE = 1.0

# Number of users in the background knowledge
S_BACKGROUND = int(ADVERSARY_RATE * NUM_OF_USERS)

# Possible sizes of background records
SAMPLE_SET = list(np.arange(start=2, stop=6, step=1))

# Source dataset
DB_FROM_READ = 'BMS-POS-full-space_10_A.data'
# DB_FROM_READ = 'loc_570_c1000_n530_PRD.dat'
# DB_FROM_READ = 'kosarak_10_A.data'
# DB_FROM_READ = 'msnbc_10_A.data'

DATASET = 'dataset.dat'


# LOCATIONAL DATA SPECIFIC VARIABLES

# Size of anonymized records ONLY in case of locational data
DOC_LEN = 20

# Size of map
MAP_HEIGHT = 30
MAP_WIDTH = 19

# Side length of each cell
CELL_SIZE = 1000.0
