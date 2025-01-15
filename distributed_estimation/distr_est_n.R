################################################################################
# Distributed M-Estimator (Huber Estimator) Simulation Study III:
# Robust estimation: Effect of local sample size n
#
# This script demonstrates the behavior of a distributed M-estimator (Huber 
# estimator) under various configurations of local sample size. It compares 
# the performance of the distributed Huber estimator to the global OLS estimator 
# and the global M-estimator (Huber estimator) in a distributed context.
#
# Note: While this example is designed to run on a local machine, you may need 
# cluster computing resources for large-scale computations or high-dimensional 
# setups. The code below demonstrates parallel computing using the `snowfall` 
# package on a local machine.
################################################################################

# Load required libraries
library(snowfall)
library(mvtnorm)  # For multivariate normal distribution
library(EnvStats)  # For Pareto distribution (rpareto)

# Check available CPU cores
available_cores <- parallel::detectCores()
cat("Available Cores:", available_cores, "\n")

# Initialize parallel computing with (available_cores - 2) cores
sfInit(parallel = TRUE, cpus = available_cores - 2)
sfLibrary(mvtnorm)  # Ensure mvtnorm is loaded on all worker nodes
sfLibrary(EnvStats)  # # Ensure EnvStats is loaded on all worker nodes

################################################################################
# Parameter Setup and Dependencies
################################################################################

# Exported variables and functions
source("../utils/functions_estimation.R")  # Adjust path as needed
n_list=c(500,1000,2500,5000,10000)  # local sample size n list
type_name=c('ols_global','Huber_global','Huber_distr_dc')  # estimator type list

################################################################################
# Wrapper Function for Simulations
# Runs distributed Huber estimators for different configurations of local sample 
# size.
################################################################################

wrapper<- function(seed){
  
  # Seed for reproducibility
  set.seed(seed)
  
  # Results container
  FResults <- list()
  count <- 1
  
  # Simulation parameters
  d=50
  N=10^5
  beta.true = rep(1,d)
  beta0.true = 0
  tau0=1

  ###############################################################################################
  #  Simulate data (X,y) based on different local sample size scenarios
  ###############################################################################################
  
  # Loop over local sample size list
  for (n in n_list)
  {
    # Specify number of machines: m, contamination proportion: epsilon, and iteration #: n_iter
    m=as.integer(N/n)
    epsilon=1/sqrt(N)
    n_iter=as.integer(log(m)+1)
    # Simulate data
    X = rmvnorm(n = N, mean = rep(0, d), sigma = diag(1,d))
    y = beta0.true + X%*%beta.true 
    X = cbind(rep(1,N),X)
    cont_idx=rbinom(N,1,epsilon)  
    # Contaminated response (we use F = Centralized Lognorm(0,1), G = 10*X1 here)
    y=y+(1-cont_idx)*(rlnorm(N)-exp(0.5))+cont_idx*(10*X[,2])
    
    ###############################################################################################
    #  Initializer for distributed Huber estimator: average of m local Huber estimators
    ###############################################################################################      
    
    index = function(c) ((c-1)*n+1):(c*n) # Helper function to decide index ranges for each local sample
    beta_dc=rep(0,d+1)    
    for (i in c(1:m))
    {
      yi=y[index(i)] 
      Xi=X[index(i),]
      ### Adaptively select tau using local sample data from the i-th machine
      tau_const_i=tau_adapt0(yi,Xi,tau0)
      ### Obtain local Huber estimator from the i-th machine
      beta_dc_i = Huber_GDBB(yi,Xi, tau_const = tau_const_i)
      beta_dc = beta_dc+beta_dc_i
    }
    beta_dc=beta_dc/m
    ### l2 estimator error (without intercept)
    err_dc=sqrt(sum((beta_dc[-1] - beta.true)^2))
    
    ###############################################################################################
    #  Distributed Huber estimator 
    ###############################################################################################  
    
    ### Step 1: Use the local sample data from the first (master) machine to adaptively select tau
    y1=y[index(1)] 
    X1=X[index(1),]
    tau_const=tau_adapt0(y1,X1,tau0)
    
    ### Step 2: Obtain distributed Huber estimator with the above tau and the above initializer
    beta.dis_dc = Distributed_Huber(y, X, m, n.iter = n_iter,intial=T, beta0 = beta_dc, tau_const = tau_const)
    l=length(beta.dis_dc)
    beta_hat_dc=beta.dis_dc[[l-1]]
    err_hat_dc=sqrt(sum((beta_hat_dc[-1] - beta.true)^2))
    
    ###############################################################################################
    #  Global Huber estimator and global OLS estimator
    ###############################################################################################   
    
    ### Global Huber estimator
    tau_const_global = tau_adapt0(y,X,tau0)
    beta_global = Huber_GDBB(y,X, tau_const = tau_const_global)
    err_global=sqrt(sum((beta_global[-1] - beta.true)^2))
    
    ### Global OLS estimator
    beta_ols_global=as.vector(solve(t(X)%*%X/N)%*%(t(X)%*%y)/N)
    err_ols_global=sqrt(sum((beta_ols_global[-1] - beta.true)^2))
    
    ###############################################################################################
    #  Collect results
    ###############################################################################################   
    
    FResults0=c(err_ols_global,err_global,err_hat_dc)
    FResults[[count]]=FResults0
    count=count+1
  }
  return (FResults)
} 


#################################################################################
#  Repeatedly run the wrapper function 100 times:
#################################################################################
nSim <- 100
#################################################################################
#  Exporting the data and functions needed by the worker nodes:
#################################################################################
sfExport("n_list", "grad", "tau_adapt0", "Huber_GDBB", "Distributed_Huber", "wrapper")
FinalResult <- sfLapply(1:nSim,wrapper)

### Write the final results into a table
iteration=rep(c(1:nSim),each=length(n_list)*length(type_name))
n=rep(rep(n_list,each=length(type_name)),nSim)
type=rep(type_name,nSim*length(n_list))
err=unlist(FinalResult)
df=data.frame(iteration,n,type,err)

### save table
write.csv(df, file = "results/output_n.csv")
sfStop()
