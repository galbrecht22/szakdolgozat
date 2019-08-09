from ot import emd2
from Hungarian import HungarianAlgorithm, VectorInt, VectorVectorDouble
import math
from functions import *
from properties import *
import h5py
from multiprocessing import Pool, Lock


# D and O values of the adversary models
# These are the parameters for function @reversing, which executes the attack.
arg_arr = [(2, 9), (5, 8), (8, 7)]


def reversing(d, o):

    # Function for generating indices from cell ID
    def id_to_index(attribute_id):
        """ h: index of row, w: index of column """
        h = (attribute_id - 1) / MAP_WIDTH
        w = (attribute_id - 1) % MAP_WIDTH
        return (h, w)

    # Function for computing physical distance between cells
    def ground_distance(cell_a, cell_b):
        index_a = id_to_index(cell_a)
        index_b = id_to_index(cell_b)
        """
        distance = sqrt(a^2 + b^2), where
        a = (x1-x2)*CELL_SIZE and
        b = (y1-y2)*CELL_SIZE
        """
        distance = math.sqrt(
            ((index_a[0] - index_b[0]) * CELL_SIZE) ** 2 + ((index_a[1] - index_b[1]) * CELL_SIZE) ** 2)
        distance = round(distance, 3)
        return distance

    # Distortion executer function
    def distort(cell, opacity):
        rec = distance_matrix[cell - 1]
        rec = [(k, v) for k, v in enumerate(rec)]
        rec.sort(key=lambda tup: tup[1], reverse=False)
        abs_dist = rec[len(key_indices)-1][1]
        scale = opacity * abs_dist
        item = next(x for x in rec if x[1] >= scale)[0]
        return item

    # Function for testing accuracy of targeted attacks
    def top_k_results(k, w, th):
        threshold = round(thresholding(w, th), 3)
        w = [[threshold if val > threshold else val for val in rec] for rec in w]
        tp_res = []
        fp_res = []
        undef_res = []
        for ik in range(k):
            correct = 0
            wrong = 0
            precarious = 0
            for js in range(S_BACKGROUND):
                rec = list(w[js])
                rec = [(k, v) for k, v in enumerate(rec)]
                rec.sort(key=lambda tup: tup[1], reverse=False)
                drec = dict(rec)
                rec_arr = [k for k, v in rec[:ik + 1]]
                if rec[0][1] == threshold:
                    precarious += 1
                elif js in rec_arr:
                    if drec[js] == threshold:
                        wrong += 1
                    else:
                        correct += 1
                else:
                    wrong += 1
            if correct + wrong == 0.0:
                fp_res.append(0.0)
                tp_res.append(0.0)
            else:
                tp_res.append(round(float(correct) / float(correct + wrong) * 100, 1))
                fp_res.append(round(float(wrong) / float(correct + wrong) * 100, 1))
            undef_res.append(round(float(precarious) / float(S_BACKGROUND) * 100, 1))
        return tp_res, fp_res, undef_res

    # Function for thresholding computed distances
    def thresholding(emdm, tr):
        # all_distances = sorted([dist for rec in emdm for dist in rec], reverse=True)[
        #      int(NUM_OF_USERS * S_BACKGROUND * (0.1 / 2)):int(
        #          NUM_OF_USERS * S_BACKGROUND * (1 - (0.1 / 2)))]
        all_distances = sorted([dist for rec in emdm for dist in rec], reverse=True)
        threshold_rate = float(tr)/100
        min_elem = min(all_distances)
        max_elem = max(all_distances)
        return min_elem + (max_elem - min_elem) * threshold_rate

    # Declaration of distortion (D) and opacity (O) rate
    D_RATE = float(d) / 10
    OPACITY = float(o) / 10

    """ ------------------------------------------------------------------------------------------------------- """
    """ -------------------------------------------START OF PROCESS ------------------------------------------- """
    """ ------------------------------------------------------------------------------------------------------- """

    # Define sorted list of all possible attributes
    key_indices = np.arange(1, MAP_HEIGHT * MAP_WIDTH + 1)

    """ ------------------------------------------ 1. SCANNING PHASE ------------------------------------------ """
    t_START = time()

    t = time()

    # Background knowledge: consists of user records, which are observed by adversary
    background_knowledge = []

    # Anonymized dataset: consists of all user's records
    anonymized_dataset = []

    # Read records from preprocessed data
    with h5py.File("matrices.h5", "r") as hf:
        obtained_records = hf.require_group('obtained')
        anonymized_records = hf.require_group('anonymized')
        keys = hf['keys'][:]
        for i in obtained_records.values():
            background_knowledge.append(i[:])
        for j in anonymized_records.values():
            anonymized_dataset.append(j[:])

    # Define set of all occurring attributes
    keys = map(lambda element: int(element), keys)

    # Map all attributes to integers
    anonymized_dataset = [map(lambda elem: int(elem), record) for record in anonymized_dataset]
    background_knowledge = [map(lambda elem: int(elem), record) for record in background_knowledge]

    list_of_keys = list(keys)
    list_of_keys.sort()

    """ ---------------------------------- 2. DISTANCE MATRIX COMPUTING PHASE ---------------------------------- """
    t = time()

    distance_matrix = [[ground_distance(p+1, q+1) for q in range(len(key_indices))] for p in range(len(key_indices))]

    rt_DIST = round(time() - t, 3)

    """ ----------------------------------------- 3. DISTORTION PHASE ----------------------------------------- """

    for i in range(np.size(background_knowledge, 0)):
        DISTORTION = int(D_RATE * len(background_knowledge[i]))
        for j in range(DISTORTION):
            background_knowledge[i][j] = distort(background_knowledge[i][j], 1 - OPACITY)

    """ ------------------------------------- 4. HISTOGRAM CREATING PHASE ------------------------------------- """

    bins = np.array(key_indices)
    bins = np.append(bins, max(bins) + 1)

    # Generate the anonymized and background histograms of each user
    anonymized_histograms = [np.histogram(record, bins=bins, density=True)[0] for record in anonymized_dataset]
    background_histograms = [np.histogram(record, bins=bins, density=True)[0] for record in background_knowledge]

    """ --------------------------------------- 5. EMD COMPUTING PHASE ---------------------------------------- """

    distance_matrix = np.array(distance_matrix)

    """
    connection_matrix -- Matrix of distances between background and anonymized histograms.
               Shape: ( M(BACKGROUND_USERS) x N(ANONYMIZED_USERS) )
        For example, weights[0][1] will represent the distance between
        the background histogram of user#1 and the anonymized histogram of user#2.
    """
    # Compact code without logging:
    # connection_matrix = [[emd(first_histogram=obtained, second_histogram=anonymized, distance_matrix=distance_matrix)
    #                for anonymized in anonymized_dataset] for obtained in background_knowledge]

    connection_matrix = [
        [0 for _ in range(np.size(anonymized_dataset, 0))] for _ in range(np.size(background_knowledge, 0))]

    count = 0
    n = 1

    t = time()

    emd_time = time()
    for i in range(np.size(background_knowledge, 0)):
        # print '...EMD computing of user {}/{}...'.format(i+1, S_BACKGROUND)
        for j in range(np.size(anonymized_dataset, 0)):
            count += 1
            if (float(count)/float(S_BACKGROUND*NUM_OF_USERS)) * 100 >= n*10:
                # print 'Processing status: {}% ( {}s )'.format(n*10, round(time() - emd_time, 3))
                n += 1
                emd_time = time()

            connection_matrix[i][j] = emd2(background_histograms[i], anonymized_histograms[j], distance_matrix)

    rt_EMD = round(time() - t, 2)

    """ ------------------------------------------ 6. MATCHING PHASE ------------------------------------------ """

    # Function for matching histograms via executing Hungarian algorithm on emd matrix
    # Distances are being restricted by threshold value
    def matching(matrix, th):
        hungarian_algorithm = HungarianAlgorithm()
        """
        assignment -- Output vector of indices.
            In a correct way of matching, every element is equal with its index,
            like every user's anonymized histogram must correlate best with
            its obtained histogram.
        """
        assignment = VectorInt(np.array([]))

        threshold = thresholding(matrix, th)
        matrix = [[threshold if val > threshold else val for val in rec] for rec in matrix]
        solution = hungarian_algorithm.Solve(VectorVectorDouble(matrix), assignment)
        fp_cases = 0
        tp_cases = 0
        undef_cases = 0
        try:
            for x in range(len(assignment)):
                if matrix[x][assignment[x]] == threshold:
                    undef_cases += 1
                elif x == assignment[x]:
                    tp_cases += 1
                else:
                    fp_cases += 1
        except IndexError:
            'Index out of bounds.'
        return tp_cases, fp_cases, undef_cases, threshold

    S_INTERVAL = str(min(SAMPLE_SET)) + '_' + str(max(SAMPLE_SET))
    DB = DB_FROM_READ.split('.', 1)[0]

    t = time()
    for th in range(1, 51, 1):
        TP, FP, UNDEF, threshold = matching(connection_matrix, th)
        KNN_TP, KNN_FP, KNN_UNDEF = top_k_results(1, connection_matrix, th)
        one_nearest = KNN_TP[0]

        """ ----------------------------------------------- SUMMARY ----------------------------------------------- """

        total = S_BACKGROUND

        if TP + FP == 0.0:
            accuracy = 0.0
            failed = 0.0

        else:
            accuracy = round((float(TP) / float(TP + FP)) * 100, 1)
            failed = round((float(FP) / float(TP + FP)) * 100, 1)

        undef = round((float(UNDEF) / float(total)) * 100, 1)

        t_END = time()
        rt_FULL = round(t_END - t_START, 2)

        lock.acquire()

        with open('model' + str(d) + '_mass_raw.txt', 'a') as logfile:
            logfile.write('%3s\t%s\t%3d\t%1.1f\t%4s\t%1.1f\t%1.1f\t%1.2f\t'
                          '%5.3f\t%3.1f\t%3.1f\t%3.1f\t%3.1f\t%4.2f\t%4.2f\n'
                          % (DB, 'EMD', NUM_OF_USERS, ADVERSARY_RATE, S_INTERVAL, D_RATE, OPACITY, float(th) / 100,
                             threshold, one_nearest, accuracy, failed, undef, rt_EMD, rt_FULL))

        with open('model' + str(d) + '_mass_raw.csv', 'a') as logfile:
            s = '%s;%s;%d;%.1f;%s;%.1f;%.1f;%.2f;%.3f;%.1f;%.1f;%.1f;%.1f;%.2f;%.2f\n' \
                % (DB, 'EMD', NUM_OF_USERS, ADVERSARY_RATE, S_INTERVAL, D_RATE, OPACITY, float(th) / 100, threshold,
                   one_nearest, accuracy, failed, undef, rt_EMD, rt_FULL)
            s = s.replace('.', ',')
            logfile.write(s)

        sresults = [str(a).replace('.', ',') for a in KNN_TP]
        with open('model' + str(d) + '_targ_tpr_raw.txt', 'a') as logfile:
            logfile.write('%s\t\t%3d\t\t%1.1f\t\t%1.1f\t\t%1.1f\t\t%.2f\t\t' %
                          ('EMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write('\t'.join(sresults))
            logfile.write('\t%3.1f\n' % accuracy)

        with open('model' + str(d) + '_targ_tpr_raw.csv', 'a') as logfile:
            logfile.write('%s;%d;%.1f;%.1f;%.1f;%.2f;' %
                          ('EMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write(';'.join(sresults))
            logfile.write(';%s;%s\n' % (str(accuracy).replace('.', ','), DB))

        sresults = [str(a).replace('.', ',') for a in KNN_UNDEF]
        with open('model' + str(d) + '_targ_undef_raw.txt', 'a') as logfile:
            logfile.write('%s\t\t%3d\t\t%1.1f\t\t%1.1f\t\t%1.1f\t\t%.2f\t\t' %
                          ('EMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write('\t'.join(sresults))
            logfile.write('\t%3.1f\n' % accuracy)

        with open('model' + str(d) + '_targ_undef_raw.csv', 'a') as logfile:
            logfile.write('%s;%d;%.1f;%.1f;%.1f;%.2f;' %
                          ('EMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write(';'.join(sresults))
            logfile.write(';%s;%s\n' % (str(accuracy).replace('.', ','), DB))

        lock.release()


def wrapf(arr):
    return reversing(*arr)


def init(l):
    global lock
    lock = l


if __name__ == '__main__':
    lck = Lock()
    p = Pool(processes=3, initializer=init, initargs=(lck,))
    print p.map(wrapf, arg_arr)
    p.close()
    p.join()
