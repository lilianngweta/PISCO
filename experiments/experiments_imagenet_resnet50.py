import numpy as np
from numpy import linalg as LA
import torch
from numpy import load
import sys, json
from itertools import product
from sklearn.metrics import top_k_accuracy_score

from logistic_regression import LogisticRegression, train_model, model_eval
from logistic_regression import LogisticRegression_bin, train_model_bin, model_eval_bin



# call this function to get experiments results for different parameters    
def get_exp_results(alpha = 1.0, seed=0, lamda=1, extractor='resnet', transf_type='woman_sketch', dataset='imagenet', 
                    eta=0.95, epochs=30,batch_size=128,learning_rate=0.001):
    
    np.random.seed(seed)
    
    # Load saved image features
    z_train_og = load('./data/Z_train_og_imagenet_resnet.npy')
    z_test_og = load('./data/Z_test_og_imagenet_resnet.npy')
    z_test_t = load('./data/Z_test_'+transf_type+'_'+dataset+'_'+extractor+'.npy')

    y_train_og = load('./data/train_og_labels_imagenet.npy',allow_pickle=True)

    y_test_og = load('./data/test_og_labels_imagenet.npy',allow_pickle=True)
    y_test_t = load('./data/test_'+transf_type+'_labels_imagenet.npy',allow_pickle=True)
    
  
    
    y_train_og = np.hstack(y_train_og)
    y_test_og = np.hstack(y_test_og)
    y_test_t = np.hstack(y_test_t)
    
    input_dim1 = z_train_og.shape[1] #number of features/dimensions on incoming data
    output_dim1 = 1000 #number of classes on ImageNet
    
    
    #Shuffle the data
    rnd_state0 = np.random.get_state()
    np.random.shuffle(z_train_og)
    np.random.set_state(rnd_state0)
    np.random.shuffle(y_train_og)  
    
    # Trained on original baseline features, tested on transformed features - no spurious correlations in the data
    lr_model_baseline = LogisticRegression(input_dim1, output_dim1)
    lr_model_baseline = train_model(torch.from_numpy(z_train_og), torch.from_numpy(y_train_og), epochs, 
                                    batch_size, learning_rate, lr_model_baseline)
    baseline_og_accuracy1, baseline_og_t5_acc1 = model_eval(torch.from_numpy(z_test_og),torch.from_numpy(y_test_og), 
                                                            batch_size,lr_model_baseline)
    
    baseline_transf_accuracy2, baseline_transf_t5_acc2 = model_eval(torch.from_numpy(z_test_t),
                                                                    torch.from_numpy(y_test_t), batch_size,lr_model_baseline)

    
    
    z_train_og_ = load('./data/Z_train_og_imagenet_resnet.npy')
    z_train_sketch_ = load('./data/Z_train_dog_sketch_imagenet_resnet.npy')
    z_train_picasso_ = load('./data/Z_train_picasso_dog_imagenet_resnet.npy')
    
    y_train_og_ = load('./data/train_og_labels_imagenet.npy',allow_pickle=True) 
    y_train_sketch_ = load('./data/train_dog_sketch_labels_imagenet.npy',allow_pickle=True) 
    y_train_picasso_ = load('./data/train_picasso_dog_labels_imagenet.npy',allow_pickle=True)
    
    y_train_og_ = np.hstack(y_train_og_)
    y_train_sketch_ = np.hstack(y_train_sketch_)
    y_train_picasso_ = np.hstack(y_train_picasso_)
       
    
    z_train_og_sketch = np.concatenate((z_train_og_, z_train_sketch_), axis=0)
    sketch_train_labels = np.concatenate((np.zeros(len(z_train_og_)), np.ones(len(z_train_sketch_))), axis=None)
    
    z_train_og_picasso = np.concatenate((z_train_og_, z_train_picasso_), axis=0)
    picasso_train_labels = np.concatenate((np.zeros(len(z_train_og_)), np.ones(len(z_train_picasso_))), axis=None)
    
    input_dim0 = z_train_og.shape[1]
    output_dim0 = 1
    
    #Shuffle the data
    rng_state = np.random.get_state()
    np.random.shuffle(z_train_og_sketch)
    np.random.set_state(rng_state)
    np.random.shuffle(sketch_train_labels)    
     
    # Get prediction coeficients of styles
    
    lr_model_sketch = LogisticRegression_bin(input_dim0, output_dim0)
    lr_model_sketch = train_model_bin(torch.from_numpy(z_train_og_sketch),
                                  torch.from_numpy(sketch_train_labels).reshape(-1,1), 
                                  epochs,batch_size, learning_rate, lr_model_sketch)
    sketch_coefficients = list(lr_model_sketch.parameters())[0].reshape(-1,1).detach().numpy()
    P1 = sketch_coefficients / np.linalg.norm(sketch_coefficients)    
    sketch_accuracy = model_eval_bin(torch.from_numpy(z_train_og_sketch),
                                     torch.from_numpy(sketch_train_labels).reshape(-1,1),
                                     batch_size,lr_model_sketch)
    print("Sketch Style Prediction accuracy")
    print(sketch_accuracy)
    
 
    #Shuffle the data
    rng1_state = np.random.get_state()
    np.random.shuffle(z_train_og_picasso)
    np.random.set_state(rng1_state)
    np.random.shuffle(picasso_train_labels)  
    
    lr_model_picasso = LogisticRegression_bin(input_dim0, output_dim0)
    lr_model_picasso = train_model_bin(torch.from_numpy(z_train_og_picasso), 
                                   torch.from_numpy(picasso_train_labels).reshape(-1,1), 
                                   epochs, batch_size, learning_rate, lr_model_picasso)
    picasso_coefficients = list(lr_model_picasso.parameters())[0].reshape(-1,1).detach().numpy()
    P2 = picasso_coefficients / np.linalg.norm(picasso_coefficients)  
    picasso_accuracy = model_eval_bin(torch.from_numpy(z_train_og_picasso),
                                      torch.from_numpy(picasso_train_labels).reshape(-1,1), 
                                      batch_size,lr_model_picasso)
    print("picasso Style Prediction accuracy")
    print(picasso_accuracy)
      

        
        
    # Find P, get post-processed features, and perform predictions
    
    # ####################### For computations below, if running on a computer that doesn't #########################
    ############   have a large memory, rewrite the next few lines to load the matrices as  #########################
    ############   subsets of rows/columns.                                                 #########################
     
    delta_z_matrix1 = z_train_og_ - z_train_sketch_ 
    delta_z_matrix2 = z_train_og_ - z_train_picasso_
    
    combined_delta_z_matrix = np.concatenate((delta_z_matrix1, delta_z_matrix2), axis=0)
    
    z_train_og_2_transfs = np.concatenate((z_train_og_, z_train_sketch_, z_train_picasso_), axis=0)
    
    k = int(z_train_og_2_transfs.shape[1]*eta) # % of original number of features
    n = z_train_og_2_transfs.shape[0]
    n_delt =  combined_delta_z_matrix.shape[0]
    
    M = - z_train_og_2_transfs.T @ z_train_og_2_transfs/n + lamda * combined_delta_z_matrix.T @ combined_delta_z_matrix /n_delt 
    
    # Perform SVD to get eigenvectors and eigenvalues
    eigenvalues, eigenvectors = LA.eigh(M)

    Q = eigenvectors[:,:(k-2)]

    P = np.concatenate((P1,P2,Q), axis=1)
        
    
    
    # Obtain post-processed features
    f_train_og = z_train_og @ P  
    f_test_og = z_test_og @ P 
    f_test_t = z_test_t @ P 
    
    
    input_dim2 = f_train_og.shape[1]-2 #number of features/dimensions of incoming data
    output_dim2 = 1000 #number of classes on ImageNet     
    
    # trained on original post-processed features, tested on transformed post-processed features 
    # without style features - no spurious correlations     
    lr_model_pisco = LogisticRegression(input_dim2, output_dim2)
    lr_model_pisco = train_model(torch.from_numpy(f_train_og[:,2:]).type(torch.FloatTensor), torch.from_numpy(y_train_og), 
                                 epochs, batch_size, learning_rate, lr_model_pisco)
    pisco_og_accuracy1, pisco_og_t5_acc1 = model_eval(torch.from_numpy(f_test_og[:,2:]).type(torch.FloatTensor),
                                                      torch.from_numpy(y_test_og), batch_size,lr_model_pisco)   
    pisco_transf_accuracy2, pisco_transf_t5_acc2 = model_eval(torch.from_numpy(f_test_t[:,2:]).type(torch.FloatTensor),
                                                              torch.from_numpy(y_test_t), batch_size,lr_model_pisco)
    
    # put all the results in a dictionary
    results_log = {}
    results_log['dataset'] = dataset
    results_log['extractor'] = extractor
    results_log['transformation_type'] = transf_type
    results_log['lamda'] = lamda
    results_log['eta'] = eta
    results_log['seed'] = seed    
    results_log['baseline_indist_acc_no_sp_corr'] = baseline_og_accuracy1
    results_log['baseline_ood_acc_no_sp_corr'] = baseline_transf_accuracy2    
    results_log['baseline_indist_t5_acc_no_sp_corr'] = baseline_og_t5_acc1
    results_log['baseline_ood_t5_acc_no_sp_corr'] = baseline_transf_t5_acc2 
    results_log['PISCO_indist_acc_no_sp_corr'] = pisco_og_accuracy1
    results_log['PISCO_ood_acc_no_sp_corr'] = pisco_transf_accuracy2      
    results_log['PISCO_indist_t5_acc_no_sp_corr'] = pisco_og_t5_acc1
    results_log['PISCO_ood_t5_acc_no_sp_corr'] = pisco_transf_t5_acc2 
    
    return results_log   


if __name__ == "__main__":
    datasets = ['imagenet'] 
    extractors= ['resnet']  
    transf_types = ['woman_sketch', 'dog_sketch', 'picasso_sp', 'picasso_dog']   
    lamdas= [1,10,50]
    etas = [0.90,0.93,0.95,0.98,1.0]
    epochs = [50]
    batch_sizes = [32768] 
    learning_rates = [0.0001]
    
    grid = list(product(datasets, extractors, transf_types, lamdas,etas,epochs,batch_sizes,learning_rates))

    i = int(float(sys.argv[1]))
   
    dataset, extractor, transf_type, lamda, eta,epochs_,batch_size,learning_rate = grid[i]
    
    results_log = get_exp_results(lamda=lamda, extractor=extractor,transf_type=transf_type, 
                                  dataset=dataset, eta=eta,epochs=epochs_,batch_size=batch_size,
                                  learning_rate=learning_rate)
    with open(f'results/results_imagenet/summary_{i}.json', 'w') as fp:
        json.dump(results_log, fp)
