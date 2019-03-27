def node_log_regr(data_file, coefficients_file, site_number):

    import pandas as pd
    import numpy as np


    #data.DC<-read.table(file=DC2.data.file, sep=",",header=T)
    #Strip out header row
    #data.DC<-data.DC[,-1]
    node_data = pd.read_csv(os.environ["DATABASE_URI"], sep = ",")


    #Calculate number of subjects available in the current study
    #(by enumerating length of ID column)
    #nsubs<-length(data.DC$id)
    rows_number = len(node_data)

    #Define design matrix (matrix of covariates) to contain BMI, SNP and the
    #interaction covariate and add a column of 1s at the start for the regression constant
    #X.mat<-cbind(rep(1,nsubs),data.DC$bmi,data.DC$bmi456,data.DC$snp)
    design_matrix = pd.concat([pd.Series(np.repeat([1], [rows_number], axis = 0)),
                              node_data["bmi"], node_data["bmi456"], node_data["snp"]], axis = 1).values


    #Load the current value of the beta vector (vector of regression coefficients) from its
    #location on the AC computer (stored during activation of block 2 of R code)
    #load(file=paste(AC.Directory,"beta.vect.next.RData",sep=""))

    beta_next = pd.read_csv(coefficients_file, header=None, sep ="\t").values

    # Use this current value of the beta vector to calculate elements from the current study
    #beta.vect<-beta.vect.next
    beta = beta_next
    # Calculate linear predictors from observed covariate values and elements of
    # current beta vector
    #lp.current<-beta.vect[1]+beta.vect[2]*X.mat[,2]+beta.vect[3]*X.mat[,3]+ beta.vect[4]*X.mat[,4]
    linear_predictor = np.dot(design_matrix, beta)


    # Apply inverse logistic transformation
    #mu.current<-exp(lp.current)/(1+exp(lp.current))
    predictor_exp = np.exp(linear_predictor)
    mu = predictor_exp/(1 + predictor_exp)

    # Derive variance function and diagonal elements for weight matrix (using squared
    # first differential of link function)
    #var.i<-(mu.current*(1-mu.current))
    #g2.i<-(1/(mu.current*(1-mu.current)))^2
    #W.mat<-diag(1/(var.i*g2.i))
    variance = mu * (1 - mu)
    link_function_differential = 1 / (variance)
    weight_matrix = np.diagflat(variance)

    #Calculate information matrix
    #info.matrix<-t(X.mat)%*%W.mat%*%X.mat
    info_matrix = np.dot(np.dot(design_matrix.transpose(), weight_matrix), design_matrix)

    #Derive u terms for score vector
    #u.i<- (data.DC$CC-mu.current)* (1/(mu.current*(1-mu.current)))
    outcome = np.matrix(node_data["CC"].values).T
    u_terms = np.multiply((outcome - mu), link_function_differential)

    #Calculate score vector
    #score.vect<-t(X.mat)%*%W.mat%*%u.i
    score_vect = np.dot(np.dot(design_matrix.T, weight_matrix), u_terms)

    #Calculate log likelihood and deviance contribution for current study
    #For convenience, ignore the element of deviance that relates to the full saturated
    # model, because that will cancel out in calculating the change in deviance from one
    # iteration to the next (Dev.total – Dev.old [see below]) because the element relating
    # to the saturated model will be the same at every iteration).
    #log.L<-sum(data.DC$CC*log(mu.current) + (1-data.DC$CC)*log(1-mu.current))
    log_likelihood = outcome.T * np.log(mu) + (1 - outcome).T * np.log(1 - mu)
    deviance = -2 * log_likelihood
    np.savetxt('C:/project/data/AC/info_matrix' + str(site_number) + ".csv", info_matrix)
    np.savetxt('C:/project/data/AC/score_vector' + str(site_number) + ".csv", score_vect)
    np.savetxt('C:/project/data/AC/deviance' + str(site_number) + ".csv", deviance)
    np.savetxt('C:/project/data/AC/samples' + str(site_number) + ".csv", [rows_number])
    return info_matrix, score_vect, deviance, rows_number


data_file = "C:/project/data/Study.1.csv"
coefficients_file = "C:/project/data/AC/beta.vect.next.csv"
for site_number in range(1, 7):
    data_file = "C:/project/data/Study." + str(site_number) + ".csv"
    node_log_regr(data_file = data_file, coefficients_file = coefficients_file,
                  site_number = site_number)

