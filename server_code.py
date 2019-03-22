import numpy as np


def calclulate_beta(matrices, scores, deviances, samples):
    info_matrix = np.empty([4, 4])
    for matrix in matrices:
        info_matrix += matrix
    score_vector = np.empty(4)
    for score in scores:
        score_vector += score
    deviance = 0
    for site_deviance in deviances:
        deviance += site_deviance
    rows = 0
    for sample in samples:
         rows += sample

    variance_covariance_matrix = np.linalg.inv(info_matrix)
    new_beta =  variance_covariance_matrix * np.asmatrix(score_vector).T
    print(score_vector)
    print(variance_covariance_matrix)
    print(new_beta)
    return 0


matrices = []
scores = []
deviances = []
samples = []
for node in range(1, 7):
    matrix = np.loadtxt('C:/project/data/AC/info_matrix' + str(node) + ".csv")
    matrices.append(matrix)
    score = np.loadtxt('C:/project/data/AC/score_vector' + str(node) + ".csv")
    scores.append(score)
    deviance = np.loadtxt('C:/project/data/AC/deviance' + str(node) + ".csv")
    deviances.append(deviance)
    sample = np.loadtxt('C:/project/data/AC/samples' + str(node) + ".csv")
    samples.append(sample)
calclulate_beta(matrices, scores, deviances, samples)
