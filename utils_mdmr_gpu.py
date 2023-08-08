import pandas as pd
import os
import numpy as np
from inspect import isclass
from joblib import Parallel, delayed
import torch

def verbose(chr,mode = 0):
    if mode == 1:
        print(chr)
    else:
        pass
def check_symmetric(X):
    """
    check whether input matrix is symmetric or not
    """ 
    if not torch.all(X==X.T):
        raise AttributeError("Distance matrix is not symmetric")
def gower_GPU(D):
    """
    Compute Gower matrix
    
    """
    
    # Dimensionality of distance matrix
    n = int(D.shape[0])

    # Create Gower's symmetry matrix (Gower, 1966)
    A = -0.5*torch.square(D)

    # Subtract column means As = (I - 1/n * 11")A
    As = A - torch.outer(torch.ones(n).cuda(),  A.mean(0))
    #Substract row
    G = As - torch.outer(As.mean(1), torch.ones(n).cuda())

    return G

def hat_matrix(X):
    """
    Calculates distance-based hat matrix for an NxM matrix of M predictors from
    N variables. Adds the intercept term for you.
    
    """
    N = X.shape[0]
    X = torch.cat((torch.ones(N)[:,None].cuda(), X),dim=-1) # add intercept
    XXT = torch.matmul(X.T, X)
    XXT_inv = XXT.inverse()
    H = torch.matmul(torch.matmul(X, XXT_inv), X.T)
    return H

def compute_SSW(H, G, df2):
    """
    Compute within group sum of squares
    
    """
    trace_HG = torch.trace(torch.matmul(H, G))
    trace_G  = torch.trace(G)

    return (trace_G - trace_HG )/df2

def compute_SSB(H, G, df1):
    """
    Compute between group sum of squares
    
    """
    
    trace_HG = torch.trace(torch.matmul(H, G))
    #print(trace_HG)
    return trace_HG/df1

def design_matrices(df):
    """
    Construct design matrix from dataframe
    
    """
    
    X_list = []
    for ii in  range(df.shape[1]):
        #TODO: category gives error
        if (df.iloc[:, ii].dtype=='object'):
            X_list.append(pd.get_dummies(df.iloc[:, ii], 
                                         drop_first=True).values)
        else:
            X_list.append(df.iloc[:, ii].values[:, np.newaxis])
            
    return X_list

class MDMR(object):
    
    def __init__(self, 
                 n_perms = None, 
                 random_state=None, 
                 n_jobs = None, 
                 verbose=0):
        
        self.n_perms = n_perms
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        
    def fit(self, X_df, D):
        
        verbose("MDMR Analyzing... ",self.verbose)
        # Check if D is symmetrical and convert to numpy
        cols = X_df.columns
        x_cpu = X_df
        D = torch.Tensor(D).double().cuda()
        X_df = torch.Tensor(np.array(pd.DataFrame(X_df))).double().cuda()
        #D = np.asarray(D)
        check_symmetric(D)
            
        # Compute Gower matrix and its trace for later
        G = gower_GPU(D)
        #print("G:\n", G)
        trG = torch.trace(G)
        
        # save variables' names
        self.vars_ = np.array(cols)
        
        # Extract list of design matrices from the dataframe. This is done
        # to be able to handle categorical features
        X_list = design_matrices(x_cpu)
    
        # Full model hat matrix
        X_full = X_df#[:,None]
        
        # N observations and m features (without intercept)
        #print(X_full.shape)
        N, m = X_full.shape
        
        H_full = hat_matrix(X_full)
        #print("H_full:\n",H_full)
        df2 = N - m
        
        verbose("Computing omnibus statistic...",self.verbose)
        den = compute_SSW(H_full, G, df2) 
        
        # Compute SSB for full model (omnibus)
        num_omni = compute_SSB(H_full, G, m)
        
        # Compute F and R2 for omnibus model
        self.F_omni_ = num_omni/den
        self.r2_omni_ = num_omni*m/trG
        self.df_omni_ = m
        #print(num_omni, den, trG, m)
        verbose("Computing individual variable statistic...",self.verbose)
        # Compute differences between H and defreees of freedom
        H_list = []
        for ii in range(len(X_list)):
            temp = X_list.copy()
            temp.pop(ii)
            if temp:
                #H_ii = H_full - hat_matrix(np.column_stack(temp))
                H_ii = H_full - hat_matrix(torch.Tensor(np.column_stack(temp)).cuda())
            else:
                H_ii = H_full 
            H_list.append(H_ii)
            
        # Compute degrees of freedom
        df1_list = []
        for X in X_list:
            m_ii = X.shape[1]
            df1_list.append(m_ii)
        
        self.df_ = np.array(df1_list)
        
        
        # Compute SSB for each column
        num_x = Parallel(n_jobs=self.n_jobs, 
                         verbose=self.verbose)(delayed(compute_SSB)(H, G, df1) for \
                                           (H, df1) in zip(H_list, df1_list))            
        num_x = torch.Tensor(num_x).cuda()
            
        self.F_ = num_x/den
        # pseudo R2.Note that we have to multiply by the degrees of freedom
        self.r2_ = torch.multiply(num_x, torch.Tensor(df1_list).cuda())/trG
        
        if self.n_perms:
            # Generate indices
            torch.cuda.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            idxs_perm = [np.random.choice(a=N, size=N, replace=False) \
                             for ii in range(self.n_perms)]
            
            verbose("Generating null model by reshuffling the distance matrix",self.verbose)
            G_perm = Parallel(n_jobs=self.n_jobs,
                              verbose=self.verbose)(delayed(gower_GPU)\
                                                  (D[idxs,:][:,idxs]) \
                                                      for idxs in idxs_perm) 
                                                    
            verbose("Computing p-value for onmibus effect...",self.verbose)
            den_perm = Parallel(n_jobs=self.n_jobs,
                                verbose=self.verbose)(delayed(compute_SSW)\
                                                    (H_full, G, df2) \
                                                        for G in G_perm)
            den_perm = torch.Tensor(den_perm).cuda()
            num_omni_perm = Parallel(n_jobs=self.n_jobs, 
                                     verbose=self.verbose)(delayed(compute_SSB)\
                                                    (H_full, G, m) \
                                                        for G in G_perm)
            num_omni_perm = torch.Tensor(num_omni_perm).cuda()
            
            self.F_perm_omni_ = num_omni_perm/den_perm
            
            verbose("computing p-value for each variable...",self.verbose)
            num_x_perm = torch.zeros((self.n_perms, len(H_list))).cuda()
            for ii, (H, df1) in enumerate(zip(H_list, df1_list)):    
                perm_temp = Parallel(n_jobs=self.n_jobs, 
                                            verbose=self.verbose)(delayed(compute_SSB)\
                                                                (H, G, df1)\
                                                                for G in G_perm)
                num_x_perm[:,ii] = torch.Tensor(perm_temp)
            
            self.F_perm_ = num_x_perm/den_perm[:, None]
            
            pval_omni_ = torch.sum(self.F_omni_ < self.F_perm_omni_)/self.n_perms
            
            pvals_ = [torch.sum(self.F_[ii] < self.F_perm_[:,ii])/self.n_perms \
                for ii in range(len(self.F_))]
            
            self.pval_omni_ = pval_omni_
            self.pvals_ = torch.Tensor(pvals_).detach().cpu().numpy()#np.array(pvals_, dtype=float)
            #self.pvals_[self.pvals_==0]=1
        else:
            self.pval_omni_ = np.nan
            self.pvals_ = np.array([np.nan for ii in range(len(self.F_))])
                                                      
        self.F_omni_ = self.F_omni_.detach().cpu().numpy()
        self.F_ = self.F_.detach().cpu().numpy()
        self.r2_omni_ = self.r2_omni_.detach().cpu().numpy()
        self.r2_ = self.r2_.detach().cpu().numpy()
        self.pval_omni_ = self.pval_omni_.detach().cpu().numpy()
        return self
    
    def summary(self):
        
        #check_is_fitted(self)
        
        summary_df = pd.DataFrame({'F': [self.F_omni_] + list(self.F_),
                                   'df': [self.df_omni_] + list(self.df_),
                                   'pseudo-R2': [self.r2_omni_] + list(self.r2_),
                                   'p-value':[self.pval_omni_] + list(self.pvals_)})
        
        summary_df.index = ['Omnibus'] + list(self.vars_)
        
        return verbose(summary_df,self.verbose)