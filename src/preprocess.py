from functions import *
import h5py
from properties import *

DB_PATH = DB_FROM_READ
select_lines(file_in=DB_PATH, file_out=DATASET, db_size=NUM_OF_USERS, shuffled=True)

with open(DATASET, 'r') as f:
    dataset = f.readlines()
    lines = [line.rstrip().split() for line in dataset]

if DB_FROM_READ[:3] == 'cab':
    offset = random.randint(0, 200 - DOC_LEN)
    print 'Offset: {}'.format(offset)
    matrix_anonymized = [[attribute for attribute in record.rstrip().split()[offset:offset+DOC_LEN]] for record in dataset]
    keys = sorted(list(set([attribute for record in matrix_anonymized for attribute in record])))
else:
    matrix_anonymized = [[attribute for attribute in record.rstrip().split()] for record in dataset]
    keys = sorted(list(set([attribute for record in matrix_anonymized for attribute in record])))

matrix_obtained = [random.sample(record, random.choice(SAMPLE_SET)) for record in matrix_anonymized[:S_BACKGROUND]]

matrix_anonymized = [np.array(record) for record in matrix_anonymized]
matrix_obtained = [np.array(record) for record in matrix_obtained]

with h5py.File("matrices.h5", "w") as hf:
    obt = hf.create_group('obtained', track_order=True)
    for i in range(np.size(matrix_obtained, 0)):
        obt.create_dataset(str(i), data=matrix_obtained[i])
    ano = hf.create_group('anonymized', track_order=True)
    for j in range(np.size(matrix_anonymized, 0)):
        ano.create_dataset(str(j), data=matrix_anonymized[j])
    hf.create_dataset('keys', data=keys)

S_INTERVAL = str(min(SAMPLE_SET)) + '-' + str(max(SAMPLE_SET))

with open('comparison_stat.txt', 'a') as logfile:
    logfile.write('%30s\t\t\t%3d\t\t%s\t\t%3d\n'
                  '----------------------------------------------------\n'
                  % (DB_FROM_READ, NUM_OF_USERS, S_INTERVAL, len(keys)))

with open('model2_mass_raw.txt', 'a') as logfile:
    logfile.write('-------------------------------------------------------------------------\n')

# print "Preprocess successful\n\tDataset: {}\n\tNumber of keys: {}\n\tNumber of users: {}\n\t" \
#       "Number of background users: {}".format(DB_FROM_READ, len(keys), NUM_OF_USERS, S_BACKGROUND)
