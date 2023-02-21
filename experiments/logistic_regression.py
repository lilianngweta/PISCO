import torch
import torchmetrics
import numpy as np
from sklearn.metrics import accuracy_score
from numpy import vstack
    
# ############ For Multi-class Classification #########################        
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_hat = self.linear(x)
        #return torch.sigmoid(y_hat) 
        return y_hat 
    
   
    
# train the model
def train_model(x_train, y_train, epochs, batch_size, learning_rate, model):
    # define the optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#lr=0.001
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs): 
        loss = None
        for index in range(0, len(x_train), batch_size):
            x_train_batch = x_train[index:index+batch_size]
            y_train_batch = y_train[index:index+batch_size]
            optimizer.zero_grad() 
            y_hat = model(x_train_batch)
            loss = criterion(y_hat, y_train_batch)
            loss.backward()
            optimizer.step()
            if (epoch)%10 == 0 or epoch==epochs-1:
                print('epoch:', epoch,',loss=',loss.item())
            if loss.item() < 0.005:  
                print('epoch:', epoch,',loss=',loss.item())
                break
        if loss.item() < 0.005:     
            break            
    return model

      
        
# evaluate the model
def model_eval(x_test, y_test, batch_size, model):
    # initialize aaccuracy metric
    accuracy1 = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=1)
    accuracy5 = torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5)
    
    
    model.eval()
    correct = 0
    total = 0
    iter_test = 0
    for index in range(0, len(x_test), batch_size):
        iter_test += 1
        x_test_batch = x_test[index:index+batch_size]
        y_test_batch = y_test[index:index+batch_size]
        
        outputs = model(x_test_batch)
        softmax = torch.nn.Softmax(dim=1)
        outputs_probs = softmax(outputs)
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += y_test_batch.size(0)

        # Total correct predictions
        correct += (predicted == y_test_batch).sum()
        acc1 = accuracy1(outputs_probs, y_test_batch)
        acc5 = accuracy5(outputs_probs, y_test_batch)

    # acc1 = 100 * (correct.item() / total)
    # acc1 = correct.item() / total
    acc1 = accuracy1.compute()
    acc5 = accuracy5.compute()
    return acc1.item(), acc5.item()
    
    
    
    
# ############ For Binary Classification #########################

class LogisticRegression_bin(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_bin, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y_hat = self.linear(x)
        return torch.sigmoid(y_hat) 
    
    
    
# train the model
def train_model_bin(x_train, y_train, epochs, batch_size, learning_rate, model):
    # define the optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)#lr=0.001
    criterion = torch.nn.BCELoss()
    
    for epoch in range(epochs): 
        loss = None
        for index in range(0, len(x_train), batch_size):
            x_train_batch = x_train[index:index+batch_size]
            y_train_batch = y_train[index:index+batch_size]
            optimizer.zero_grad() 
            y_hat = model(x_train_batch.float())
            loss = criterion(y_hat.float(), y_train_batch.float())
            loss.backward()
            optimizer.step()
            if (epoch)%10 == 0 or epoch==epochs-1:
                print('epoch:', epoch,',loss=',loss.item())
            if loss.item() < 0.005:  
                print('epoch:', epoch,',loss=',loss.item())
                break
        if loss.item() < 0.005:     
            break            
    return model

        
# evaluate the model
def model_eval_bin(x_test, y_test, batch_size, model):
    
    model.eval()
    
    predictions, actuals = list(), list()
    
    for index in range(0, len(x_test), batch_size):
        x_test_batch = x_test[index:index+batch_size]
        y_test_batch = y_test[index:index+batch_size]

        yhat = model.forward(x_test_batch.float())
        yhat = np.where(yhat.detach().numpy() < 0.5, 0, 1)
        actual = y_test_batch.numpy()
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))

        predictions.append(yhat)
        actuals.append(actual)  

    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc




