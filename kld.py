from functions import *
from properties import *
from Hungarian import HungarianAlgorithm, VectorDouble, VectorInt, VectorVectorDouble
from KL_Divergence import KL_Divergence
import h5py
from multiprocessing import Pool, Lock


# D values of the adversary models
# These are the parameters for function @reversing, which executes the attack.
arg_arr = [2, 5, 8]


def reversing(d):

    # Function for testing accuracy of targeted attacks
    def top_k_results(k, w, th):
        threshold = round(thresholding(w, th), 3)
        # print threshold
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
                # print rec
                drec = dict(rec)
                # print drec
                rec_arr = [k for k, v in rec[:ik + 1]]
                # print rec_arr
                if rec[0][1] == threshold:
                    precarious += 1
                elif js in rec_arr:
                    if drec[js] == threshold:
                        # print ik, js, drec[js], threshold
                        # precarious += 1
                        wrong += 1
                    else:
                        correct += 1
                else:
                    wrong += 1
            if correct + wrong == 0.0:
                fp_res.append(0.0)
                tp_res.append(0.0)
            else:
                # print correct, wrong, precarious
                tp_res.append(round(float(correct) / float(correct + wrong) * 100, 1))
                fp_res.append(round(float(wrong) / float(correct + wrong) * 100, 1))
            undef_res.append(round(float(precarious) / float(S_BACKGROUND) * 100, 1))
        return tp_res, fp_res, undef_res

    # Function for thresholding computed distances
    def thresholding(kldm, tr):
        # all_distances = sorted([dist for rec in emdm for dist in rec], reverse=True)[
        #      int(NUM_OF_USERS * S_BACKGROUND * (0.1 / 2)):int(
        #          NUM_OF_USERS * S_BACKGROUND * (1 - (0.1 / 2)))]
        all_distances = sorted([dist for rec in kldm for dist in rec], reverse=True)
        threshold_rate = float(tr)/100
        min_elem = min(all_distances)
        max_elem = max(all_distances)
        return min_elem + (max_elem - min_elem) * threshold_rate

    """ ---------------------------- START OF PROCESS --------------------------- """
    t_START = time()

    # Background knowledge: consists of user records, which are observed by adversary
    background_knowledge = []

    # Anonymized dataset: consists of all user's records
    anonymized_dataset = []

    t = time()

    # Read records from preprocessed data
    with h5py.File("matrices.h5", "r") as hf:
        obtained_records = hf.require_group('obtained')
        anonymized_records = hf.require_group('anonymized')
        keys = hf['keys'][:]
        for i in obtained_records.values():
            background_knowledge.append(i[:])
        for i in anonymized_records.values():
            anonymized_dataset.append(i[:])

    # Generate the sorted list of occurring attributes
    keys = map(lambda element: int(element), keys)

    # Map all attributes to integers
    for i in range(np.size(anonymized_dataset, 0)):
        anonymized_dataset[i] = map(lambda item: int(item), anonymized_dataset[i])
    for i in range(np.size(background_knowledge, 0)):
        background_knowledge[i] = list(map(lambda element: int(element), background_knowledge[i]))
    list_of_keys = list(keys)
    list_of_keys.sort()

    # Distortion executing
    D_RATE = float(d) / 10
    for i in range(np.size(background_knowledge, 0)):
        DISTORTION = int(D_RATE * len(background_knowledge[i]))
        for j in range(DISTORTION):
            background_knowledge[i][j] = random.choice(list_of_keys)

    """
    ATTRIBUTES -- Number of all occurring attributes.
    """
    ATTRIBUTES = len(list_of_keys)

    # Map attributes to indices, to produce bins properly.
    # It's important, to generate bins with unified width,
    # to get the correct distribution.
    key_indices = list(map(lambda key: list_of_keys.index(key) + 1, list_of_keys))

    # Map attributes to indices, to fit for the corresponding bins.
    anonymized_dataset = [list(map(lambda elem: list_of_keys.index(elem) + 1, record)) for record in anonymized_dataset]
    background_knowledge = [list(map(lambda elem: list_of_keys.index(elem) + 1, record)) for record in background_knowledge]

    list_of_keys = np.asarray(list_of_keys)

    bins = list(key_indices)
    bins.append(max(bins) + 1)

    # Generate the anonymized and background histograms of each user
    anonymized_histograms = [np.histogram(record, bins=bins, density=True)[0] for record in anonymized_dataset]
    background_histograms = [np.histogram(record, bins=bins, density=True)[0] for record in background_knowledge]

    print '2. Creating histograms: successful.'

    """ ------------------------------------------------------------------------- """

    """ ---------- Computing connection_matrix via Kullback-Leibler Divergence. ----------- """
    """ ------ The method was implemented in C++ to enhance running time. ------- """

    """ ------------------------------------------------------------------------- """

    """
    connection_matrix -- Matrix of distances between background and anonymized histograms.
               Shape: ( M(BACKGROUND_USERS) x N(ANONYMIZED_USERS) )
        For example, weights[0][1] will represent the distance between
        the background histogram of user#1 and the anonymized histogram of user#2.
    """
    connection_matrix = [
        [0 for _ in range(np.size(anonymized_histograms, 0))] for _ in range(np.size(background_histograms, 0))]

    klDiv = KL_Divergence()

    t = time()

    # Compact code without logging:
    # connection_matrix = [[klDiv.kl_div(VectorDouble(b_record), VectorDouble(a_record))
    #             for a_record in anonymized_dataset] for b_record in background_knowledge]

    t_w = time()
    for p in range(np.size(background_histograms, 0)):
        pk = background_histograms[p]
        c_pk = VectorDouble(pk)
        # if (p + 1) % 50 == 0:
        #     print '\t\tStatus: {}. user is under process... ( {}s )'.format(p+1, round(time()-t_w, 2))
        for q in range(np.size(anonymized_histograms, 0)):
            qk = anonymized_histograms[q]
            c_qk = VectorDouble(qk)
            connection_matrix[p][q] = round(klDiv.kl_div(c_pk, c_qk), 3)

    rt_KLD = round(time() - t, 2)

    for p in range(np.size(background_histograms, 0)):
        connection_matrix[p] = np.array(connection_matrix[p])

    """ ------------------------------------------------------------------------- """

    """ ----------- Executing user matching via Hungarian Algorithm. ------------ """
    """ ----- The algorithm was implemented in C++ to enhance running time. ----- """

    """ ------------------------------------------------------------------------- """

    t = time()

    # Function for matching histograms via executing Hungarian algorithm on emd matrix
    # Distances are being restricted by threshold value
    def matching(matrix, th):
        hungarian_algorithm = HungarianAlgorithm()
        """
        assignment -- Output vector of indexes.
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
    for th in range(2, 101, 2):
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

    # Saving results to statistical log files.
        with open('model' + str(d) + '_mass_raw.txt', 'a') as logfile:
            logfile.write('%3s\t%s\t%3d\t%1.1f\t%4s\t%1.1f\t---\t%1.2f\t'
                          '%5.3f\t%3.1f\t%3.1f\t%3.1f\t%3.1f\t%4.2f\t%4.2f\n'
                          % (DB, 'KLD', NUM_OF_USERS, ADVERSARY_RATE, S_INTERVAL, D_RATE, float(th) / 100,
                             threshold, one_nearest, accuracy, failed, undef, rt_KLD, rt_FULL))

        with open('model' + str(d) + '_mass_raw.csv', 'a') as logfile:
            s = '%s;%s;%d;%.1f;%s;%.1f;---;%.2f;%.3f;%.1f;%.1f;%.1f;%.1f;%.2f;%.2f\n' \
                % (DB, 'KLD', NUM_OF_USERS, ADVERSARY_RATE, S_INTERVAL, D_RATE, float(th) / 100, threshold,
                   one_nearest, accuracy,
                   failed, undef, rt_KLD, rt_FULL)
            s = s.replace('.', ',')
            logfile.write(s)

        sresults = [str(a).replace('.', ',') for a in KNN_TP]
        with open('model' + str(d) + '_targ_tpr_raw.txt', 'a') as logfile:
            logfile.write('%s\t\t%3d\t\t%1.1f\t\t%1.1f\t\t---\t\t%.2f\t\t' %
                          ('KLD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, float(th) / 100))
            logfile.write('\t'.join(sresults))
            logfile.write('\t%3.1f\n' % accuracy)

        with open('model' + str(d) + '_targ_tpr_raw.csv', 'a') as logfile:
            logfile.write('%s;%d;%.1f;%.1f;---;%.2f;' %
                          ('KLD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, float(th) / 100))
            logfile.write(';'.join(sresults))
            logfile.write(';%s;%s\n' % (str(accuracy).replace('.', ','), DB))

        sresults = [str(a).replace('.', ',') for a in KNN_UNDEF]
        with open('model' + str(d) + '_targ_undef_raw.txt', 'a') as logfile:
            logfile.write('%s\t\t%3d\t\t%1.1f\t\t%1.1f\t\t---\t\t%.2f\t\t' %
                          ('KLD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, float(th) / 100))
            logfile.write('\t'.join(sresults))
            logfile.write('\t%3.1f\n' % accuracy)

        with open('model' + str(d) + '_targ_undef_raw.csv', 'a') as logfile:
            logfile.write('%s;%d;%.1f;%.1f;---;%.2f;' %
                          ('KLD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, float(th) / 100))
            logfile.write(';'.join(sresults))
            logfile.write(';%s;%s\n' % (str(accuracy).replace('.', ','), DB))

        lock.release()


def init(l):
    global lock
    lock = l


if __name__ == '__main__':
    lck = Lock()
    p = Pool(processes=3, initializer=init, initargs=(lck,))
    print p.map(reversing, arg_arr)
    p.close()
    p.join()
