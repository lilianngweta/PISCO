import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy import linalg as LA
import torch
from numpy import load
import sys, json
from itertools import product
from sklearn import preprocessing



def binarize(y):    
    y = np.copy(y) > 5
    return y.astype(int)

# Function for creating spurious correlations
def create_spurious_corr(z, z_t, y_og, spu_corr= 0.1, binarize_label=False):
    y_bin = binarize(y_og)
    mod_labels = np.logical_xor(y_bin, np.random.binomial(1, spu_corr, size=len(y_bin)))
    
    modified_images = z_t[mod_labels]
    unmodified_images = z[~mod_labels]
    all_z = np.concatenate((modified_images, unmodified_images), axis=0)
    
    all_img_labels = None
    
    if binarize_label:
        modified_imgs_labels = y_bin[mod_labels]
        unmodified_imgs_labels = y_bin[~mod_labels]
        all_img_labels = np.concatenate((modified_imgs_labels, unmodified_imgs_labels), axis=None)
    else:
        modified_imgs_labels = y_og[mod_labels]
        unmodified_imgs_labels = y_og[~mod_labels]
        all_img_labels = np.concatenate((modified_imgs_labels, unmodified_imgs_labels), axis=None)    
        
    return all_z, all_img_labels 
    


# call this function to get experiments results for different parameters
def get_exp_results(alpha = 1.0, seed=0, lamda=1, extractor='mlp', transf_type='colored', 
                    dataset='mnist', eta=0.95):
    
    np.random.seed(seed)
    
    # Load saved image features
    z_train_og = load('./data/Z_train_og_mnist_mlp.npy')
    z_train_t = load('./data/Z_train_green_mnist_mlp.npy')

    z_test_og = load('./data/Z_test_og_mnist_mlp.npy')
    z_test_t = load('./data/Z_test_green_mnist_mlp.npy')

    y_train_og = load('./data/train_labels_mnist.npy')

    y_test_og = load('./data/test_labels_mnist.npy')
    
    
    # Create spurious correlations on train and test sets
    z_train, train_labels = create_spurious_corr(z_train_og, z_train_t, y_train_og, 
                                             spu_corr= alpha, binarize_label=False)

    z_test_indist, indist_test_labels = create_spurious_corr(z_test_og, z_test_t, y_test_og, 
                                                             spu_corr= alpha, binarize_label=False)

    z_test_ood, ood_test_labels = create_spurious_corr(z_test_og, z_test_t, y_test_og, 
                                                             spu_corr= 1-alpha, binarize_label=False)
    

    # concatenate original and colored features
    z_train_og_t = np.concatenate((z_train_og, z_train_t), axis=0)
    t_train_labels = np.concatenate((np.zeros(len(z_train_og)), np.ones(len(z_train_t))), axis=None) 
    z_test_og_t = np.concatenate((z_test_og, z_test_t), axis=0)
    t_test_labels = np.concatenate((np.zeros(len(z_test_og)), np.ones(len(z_test_t))), axis=None) 

    # Obtain prediction coefficients of color
    lr_model_t = LogisticRegression(random_state=0).fit(z_train_og_t, t_train_labels)
    t_coefficients = lr_model_t.coef_.reshape(-1,1)
    P1 = t_coefficients / np.linalg.norm(t_coefficients)

    # Prediction Accuracies on image features extracted using a baseline model (mlp) - spurious correlations in the data
    logistic_regression_on_baseline = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                                  random_state=0).fit(z_train,train_labels)                                                                                     
    baseline_accuracy0 = logistic_regression_on_baseline.score(z_train, train_labels)
    baseline_accuracy1 = logistic_regression_on_baseline.score(z_test_indist, indist_test_labels)
    baseline_accuracy2 = logistic_regression_on_baseline.score(z_test_ood, ood_test_labels)   
    
    # Trained on original baseline features, tested on colored features - no spurious correlations in the data
    logistic_regression_on_baseline_og = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                                     random_state=0).fit(z_train_og,y_train_og)                                                                                     
    baseline_og_accuracy0 = logistic_regression_on_baseline_og.score(z_train_og, y_train_og)
    baseline_og_accuracy1 = logistic_regression_on_baseline_og.score(z_test_og, y_test_og)
    baseline_transf_accuracy2 = logistic_regression_on_baseline_og.score(z_test_t, y_test_og)
    

    # *********** Find P, get post-processed features, and perform predictions *******************************#
    
    k = int(z_train_og_t.shape[1]*eta) # % of original number of features
    n = z_train_og_t.shape[0]

    delta_z_matrix = z_train_og - z_train_t 

    M = - z_train_og_t.T @ z_train_og_t/n + lamda * delta_z_matrix.T @ delta_z_matrix / (n // 2 ) 

    # Performing SVD to get eigenvectors and eigenvalues
    eigenvalues, eigenvectors = LA.eigh(M)

    # Forming P from eigenvectors and coeficients of color
    Q = eigenvectors[:,:(k-1)]
    P = np.concatenate((P1,Q), axis=1)
    
    # Obtain post-processed features
    f_train_og = z_train_og @ P  
    f_train = z_train @ P
    f_test_indist = z_test_indist @ P
    f_test_ood = z_test_ood @ P
    f_test_og = z_test_og @ P
    f_test_t = z_test_t @ P
    f_test_og_t = z_test_og_t @ P    

    # Classification task using all post-processed features except style features - spurious correlations  
    lr_model_pisco_sp = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                        random_state=0).fit(f_train[:,1:],train_labels)
    pisco_sp_accuracy0 = lr_model_pisco_sp.score(f_train[:,1:], train_labels)
    pisco_sp_accuracy1 = lr_model_pisco_sp.score(f_test_indist[:,1:], indist_test_labels)
    pisco_sp_accuracy2 = lr_model_pisco_sp.score(f_test_ood[:,1:], ood_test_labels)
    
    # Classification task using all post-processed features except style features - no spurious correlations   
    lr_model_pisco_no_sp = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                        random_state=0).fit(f_train_og[:,1:],y_train_og)
    pisco_no_sp_accuracy0 = lr_model_pisco_no_sp.score(f_train_og[:,1:], y_train_og)
    pisco_no_sp_accuracy1 = lr_model_pisco_no_sp.score(f_test_og[:,1:], y_test_og)
    pisco_no_sp_accuracy2 = lr_model_pisco_no_sp.score(f_test_t[:,1:], y_test_og)
    
    # put all the results in a dictionary
    results_log = {}
    results_log['dataset'] = dataset
    results_log['extractor'] = extractor
    results_log['transformation_type'] = transf_type
    results_log['alpha'] = alpha
    results_log['lamda'] = lamda
    results_log['eta'] = eta
    results_log['seed'] = seed    
    results_log['baseline_train_accuracy_sp_corr'] = baseline_accuracy0
    results_log['baseline_indist_accuracy_sp_corr'] = baseline_accuracy1
    results_log['baseline_ood_accuracy_sp_corr'] = baseline_accuracy2       
    results_log['baseline_train_acc_no_sp_corr'] = baseline_og_accuracy0
    results_log['baseline_indist_acc_no_sp_corr'] = baseline_og_accuracy1
    results_log['baseline_ood_acc_no_sp_corr'] = baseline_transf_accuracy2            
    results_log['PISCO_train_accuracy_sp_corr'] = pisco_sp_accuracy0
    results_log['PISCO_indist_accuracy_sp_corr'] = pisco_sp_accuracy1
    results_log['PISCO_ood_accuracy_sp_corr'] = pisco_sp_accuracy2    
    results_log['PISCO_train_acc_no_sp_corr'] = pisco_no_sp_accuracy0
    results_log['PISCO_indist_acc_no_sp_corr'] = pisco_no_sp_accuracy1
    results_log['PISCO_ood_acc_no_sp_corr'] = pisco_no_sp_accuracy2   
    
    return results_log


if __name__ == "__main__":
    ITERS = range(10)
    datasets = ['mnist']
    extractors= ['mlp']
    transf_types = ['colored']
    alphas = [0.5, 0.75, 0.90, 0.95, 0.99,1.0]
    lamdas= [0,1,10,50] 
    etas = [0.95]

    grid = list(product(datasets, extractors, transf_types, alphas, lamdas,etas,ITERS))
    
    i = int(float(sys.argv[1]))
    dataset, extractor, transf_type, alpha, lamda, eta, ITER = grid[i]
    

    results_log = get_exp_results(alpha = alpha, seed=ITER, lamda=lamda, extractor=extractor, 
                                  transf_type=transf_type, dataset=dataset, eta=eta)

    with open(f'results/results_mnist/summary_{i}.json', 'w') as fp:
        json.dump(results_log, fp)
        
