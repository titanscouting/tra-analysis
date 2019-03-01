#Titan Robotics Team 2022: ML Module
#Written by Arthur Lu & Jacob Levine
#Notes:
#   this should be imported as a python module using 'import titanlearn'
#   this should be included in the local directory or environment variable
#   this module has not been optimized for multhreaded computing
#   this module learns from its mistakes far faster than 2022's captains
#setup:

__version__ = "1.0.0.001"

#changelog should be viewed using print(analysis.__changelog__)
__changelog__ = """changelog:
1.0.0.xxx:
    -added generation of ANNS, basic SGD training"""
__author__ = (
    "Arthur Lu <arthurlu@ttic.edu>, "
    "Jacob Levine <jlevine@ttic.edu>,"
    )
__all__ = [
    'linear_nn',
    'train_sgd_minibatch',
    'train_sgd_simple'
    ]
#imports
import torch
import warnings
from collections import OrderedDict
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import math

#enable CUDA if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#linear_nn: creates a fully connected network given params
def linear_nn(in_dim, hidden_dim, out_dim, num_hidden, act_fn="tanh", end="none"):
    if act_fn.lower()=="tanh":
        k=OrderedDict([("in", torch.nn.Linear(in_dim,hidden_dim)), ('tanh0', torch.nn.Tanh())])
        for i in range(num_hidden):
            k.update({"lin"+str(i+1): torch.nn.Linear(hidden_dim,hidden_dim), "tanh"+str(i+1):torch.nn.Tanh()})
        
    elif act_fn.lower()=="sigmoid":
        k=OrderedDict([("in", torch.nn.Linear(in_dim,hidden_dim)), ('sig0', torch.nn.Sigmoid())])
        for i in range(num_hidden):
            k.update({"lin"+str(i+1): torch.nn.Linear(hidden_dim,hidden_dim), "sig"+str(i+1):torch.nn.Sigmoid()})

    elif act_fn.lower()=="relu":
        k=OrderedDict([("in", torch.nn.Linear(in_dim,hidden_dim)), ('relu0', torch.nn.ReLU())])
        for i in range(num_hidden):
            k.update({"lin"+str(i+1): torch.nn.Linear(hidden_dim,hidden_dim), "relu"+str(i+1):torch.nn.ReLU()})

    elif act_fn.lower()=="leaky relu":
        k=OrderedDict([("in", torch.nn.Linear(in_dim,hidden_dim)), ('lre0', torch.nn.LeakyReLU())])
        for i in range(num_hidden):
            k.update({"lin"+str(i+1): torch.nn.Linear(hidden_dim,hidden_dim), "lre"+str(i+1):torch.nn.LeakyReLU()})
    else:
        warnings.warn("Did not specify a valid inner activation function. Returning nothing.")
        return None

    if end.lower()=="softmax":
        k.update({"out": torch.nn.Linear(hidden_dim,out_dim), "softmax": torch.nn.Softmax()})
    elif end.lower()=="none":
        k.update({"out": torch.nn.Linear(hidden_dim,out_dim)})
    elif end.lower()=="sigmoid":
        k.update({"out": torch.nn.Linear(hidden_dim,out_dim), "sigmoid": torch.nn.Sigmoid()})
    else:
        warnings.warn("Did not specify a valid final activation function. Returning nothing.")
        return None
    
    return torch.nn.Sequential(k)

#train_sgd_simple: trains network using SGD
def train_sgd_simple(net, evalType, data, ground, dev=None, devg=None, iters=1000, learnrate=1e-4, testevery=1, graphsaveloc=None, modelsaveloc=None, loss="mse"):
    model=net.to(device)
    data=data.to(device)
    ground=ground.to(device)
    if dev != None:
        dev=dev.to(device)
    losses=[]
    dev_losses=[]
    if loss.lower()=="mse":
        loss_fn = torch.nn.MSELoss()
    elif loss.lower()=="cross entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss.lower()=="nll":
        loss_fn = torch.nn.NLLLoss()
    elif loss.lower()=="poisson nll":
        loss_fn = torch.nn.PoissonNLLLoss()
    else: 
        warnings.warn("Did not specify a valid loss function. Returning nothing.")
        return None
    optimizer=torch.optim.SGD(model.parameters(), lr=learnrate)
    for i in range(iters):
        if i%testevery==0:
            with torch.no_grad():
                output = model(data)
                if evalType == "ap":
                    ap = metrics.average_precision_score(ground.cpu().numpy(), output.cpu().numpy())
                if evalType == "regression":
                    ap = metrics.explained_variance_score(ground.cpu().numpy(), output.cpu().numpy())
                losses.append(ap)
                print(str(i)+": "+str(ap))
                plt.plot(np.array(range(0,i+1,testevery)),np.array(losses), label="train AP")
                if dev != None:
                    output = model(dev)
                    print(evalType)
                    if evalType == "ap":
                        
                        ap = metrics.average_precision_score(devg.numpy(), output.numpy())
                        dev_losses.append(ap)
                        plt.plot(np.array(range(0,i+1,testevery)),np.array(losses), label="dev AP")
                    elif evalType == "regression":
                        ev = metrics.explained_variance_score(devg.numpy(), output.numpy())
                        dev_losses.append(ev)
                        plt.plot(np.array(range(0,i+1,testevery)),np.array(losses), label="dev EV")

                    
                if graphsaveloc != None:
                    plt.savefig(graphsaveloc+".pdf")
        with torch.enable_grad():
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, ground)
            print(loss.item())
            loss.backward()
            optimizer.step()
    if modelsaveloc != None:
        torch.save(model, modelsaveloc)
    plt.show()
    return model

#train_sgd_minibatch: same as above, but with minibatches
def train_sgd_minibatch(net, data, ground, dev=None, devg=None, epoch=100, batchsize=20, learnrate=1e-4, testevery=20, graphsaveloc=None, modelsaveloc=None, loss="mse"):
    model=net.to(device)
    data=data.to(device)
    ground=ground.to(device)
    if dev != None:
        dev=dev.to(device)
    losses=[]
    dev_losses=[]
    if loss.lower()=="mse":
        loss_fn = torch.nn.MSELoss()
    elif loss.lower()=="cross entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif loss.lower()=="nll":
        loss_fn = torch.nn.NLLLoss()
    elif loss.lower()=="poisson nll":
        loss_fn = torch.nn.PoissonNLLLoss()
    else: 
        warnings.warn("Did not specify a valid loss function. Returning nothing.")
        return None
    optimizer=torch.optim.LBFGS(model.parameters(), lr=learnrate)
    itercount=0
    for i in range(epoch):
        print("EPOCH "+str(i)+" OF "+str(epoch-1))
        batches=math.ceil(data.size()[0].item()/batchsize)
        for j in range(batches):
            batchdata=[]
            batchground=[]
            for k in range(j*batchsize, min((j+1)*batchsize, data.size()[0].item()),1):
                batchdata.append(data[k])
                batchground.append(ground[k])
            batchdata=torch.stack(batchdata)
            batchground=torch.stack(batchground)
            if itercount%testevery==0:
                with torch.no_grad():
                    output = model(data)
                    ap = metrics.average_precision_score(ground.numpy(), output.numpy())
                    losses.append(ap)
                    print(str(i)+": "+str(ap))
                    plt.plot(np.array(range(0,i+1,testevery)),np.array(losses))
                    if dev != None:
                        output = model(dev)
                        ap = metrics.average_precision_score(devg.numpy(), output.numpy())
                        dev_losses.append(ap)
                        plt.plot(np.array(range(0,i+1,testevery)),np.array(losses), label="dev AP")
                    if graphsaveloc != None:
                        plt.savefig(graphsaveloc+".pdf")
            with torch.enable_grad():
                optimizer.zero_grad()
                output = model(batchdata)
                loss = loss_fn(output, ground)
                loss.backward()
                optimizer.step()
            itercount +=1
    if modelsaveloc != None:
        torch.save(model, modelsaveloc)
    plt.show()
    return model

def retyuoipufdyu():
    
    data = torch.tensor([[ 1.,  2.,  5.,  2.,  5.],
        [27.,  8.,  4.,  6., 10.],
        [12., 12., 12.,  5.,  6.],
        [10., 12., 10., 20.,  2.],
        [ 1.,  2.,  3.,  4.,  5.]])
    ground = torch.tensor([15., 55., 47., 54., 15.])
    model = linear_nn(5, 10, 1, 3, act_fn = "relu")
    return train_sgd_simple(model,"regression", data, ground, learnrate=1e-2)
