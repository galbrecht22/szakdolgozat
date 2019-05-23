from Hungarian import HungarianAlgorithm, VectorInt, VectorVectorDouble
from functions import *
from properties import *
import h5py
from gensim.models import KeyedVectors
from multiprocessing import Lock, Pool
import cPickle


# D and O values of the adversary models
# These are the parameters for function @reversing, which executes the attack.
arg_arr = [(2, 9), (5, 8), (8, 7)]


def reversing(d, o):
    # Function for thresholding computed distances
    def thresholding(wmdm, tr):
        # all_distances = sorted([dist for rec in emdm for dist in rec], reverse=True)[
        #      int(NUM_OF_USERS * S_BACKGROUND * (0.1 / 2)):int(
        #          NUM_OF_USERS * S_BACKGROUND * (1 - (0.1 / 2)))]
        all_distances = sorted([dist for rec in wmdm for dist in rec], reverse=True)
        threshold_rate = float(tr) / 100
        min_elem = min(all_distances)
        max_elem = max(all_distances)
        return min_elem + (max_elem - min_elem) * threshold_rate

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

    # Declaration of distortion (D) and opacity (O) rate
    D_RATE = float(d) / 10
    OPACITY = float(o) / 10

    t_START = time()

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
        for i in anonymized_records.values():
            anonymized_dataset.append(i[:])
    for i in range(np.size(background_knowledge, 0)):
        background_knowledge[i] = list(background_knowledge[i])
    for i in range(np.size(anonymized_dataset, 0)):
        anonymized_dataset[i] = list(anonymized_dataset[i])

    # Read word_vectors
    word_vectors = KeyedVectors.load("vectors_BMS.kv", mmap='r')
    list_of_keys = sorted(word_vectors.vocab)
    VOCAB = len(list_of_keys)

    with open('w2v_dist_BMS.p', 'rb') as df:
        abs_dist = cPickle.load(df)
    scale = abs_dist * (1 - OPACITY)

    # Distortion executing
    for i in range(np.size(background_knowledge, 0)):
        DISTORTION = int(D_RATE * len(background_knowledge[i]))
        for j in range(DISTORTION):
            all_words = word_vectors.most_similar(positive=background_knowledge[i][j], topn=len(list_of_keys) - 1)
            item = next((x for x in all_words if x[1] <= 1 - scale), all_words[len(list_of_keys) - 2])
            background_knowledge[i][j] = item[0]

    t = time()

    connection_matrix = [
        [0 for x in range(np.size(anonymized_dataset, 0))] for y in range(np.size(background_knowledge, 0))]

    t_wmd = time()
    for p in range(np.size(background_knowledge, 0)):
        print 'Elapsed time: {}s'.format(round(time() - t_wmd, 2))
        print '...WMD computing of user {}/{}...'.format(p + 1, S_BACKGROUND)
        for q in range(np.size(anonymized_dataset, 0)):
            connection_matrix[p][q] = word_vectors.wmdistance(background_knowledge[p], anonymized_dataset[q])

    rt_WMD = round(time() - t, 2)

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

    for th in range(2, 101, 2):
        TP, FP, UNDEF, threshold = matching(connection_matrix, th)
        KNN_TP, KNN_FP, KNN_UNDEF = top_k_results(10, connection_matrix, th)
        one_nearest = KNN_TP[0]

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
                          % (DB, 'WMD', NUM_OF_USERS, ADVERSARY_RATE, S_INTERVAL, D_RATE, OPACITY, float(th) / 100,
                             threshold, one_nearest, accuracy, failed, undef, rt_WMD, rt_FULL))

        with open('model' + str(d) + '_mass_raw.csv', 'a') as logfile:
            s = '%s;%s;%d;%.1f;%s;%.1f;%.1f;%.2f;%.3f;%.1f;%.1f;%.1f;%.1f;%.2f;%.2f\n' \
                % (DB, 'WMD', NUM_OF_USERS, ADVERSARY_RATE, S_INTERVAL, D_RATE, OPACITY, float(th) / 100, threshold,
                   one_nearest, accuracy, failed, undef, rt_WMD, rt_FULL)
            s = s.replace('.', ',')
            logfile.write(s)

        sresults = [str(a).replace('.', ',') for a in KNN_TP]
        with open('model' + str(d) + '_targ_tpr_raw.txt', 'a') as logfile:
            logfile.write('%s\t\t%3d\t\t%1.1f\t\t%1.1f\t\t%1.1f\t\t%.2f\t\t' %
                          ('WMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write('\t'.join(sresults))
            logfile.write('\t%3.1f\n' % accuracy)

        with open('model' + str(d) + '_targ_tpr_raw.csv', 'a') as logfile:
            logfile.write('%s;%d;%.1f;%.1f;%.1f;%.2f;' %
                          ('WMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write(';'.join(sresults))
            logfile.write(';%s;%s\n' % (str(accuracy).replace('.', ','), DB))

        sresults = [str(a).replace('.', ',') for a in KNN_UNDEF]
        with open('model' + str(d) + '_targ_undef_raw.txt', 'a') as logfile:
            logfile.write('%s\t\t%3d\t\t%1.1f\t\t%1.1f\t\t%1.1f\t\t%.2f\t\t' %
                          ('WMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
            logfile.write('\t'.join(sresults))
            logfile.write('\t%3.1f\n' % accuracy)

        with open('model' + str(d) + '_targ_undef_raw.csv', 'a') as logfile:
            logfile.write('%s;%d;%.1f;%.1f;%.1f;%.2f;' %
                          ('WMD', NUM_OF_USERS, ADVERSARY_RATE, D_RATE, OPACITY, float(th) / 100))
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
