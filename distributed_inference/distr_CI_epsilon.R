################################################################################
# Distributed M-Estimator (Huber Estimator) Simulation Study IV:
# Robust inference: Effect of contamination proportion and scale
#
# This script demonstrates the coverage and width of a distributed bootrap CI
# for distributed M-estimator (Huber estimator) under various configurations 
# of contamination proportion and scale. It compares the performance of 
# (1) M-boot: distributed bootstrap CI for distributed M estimator
# (2) Debias-M-boot: debiased distributed bootstrap CI for distributed M estimator
# (3) M-normal: asymptotic normality based CI for distributed M estimator
# (4) OLS-normal: asymptotic normality based CI for global OLS estimator
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
library(expm)  # For sqrt of a matrix 

# Check available CPU cores
available_cores <- parallel::detectCores()
cat("Available Cores:", available_cores, "\n")

# Initialize parallel computing with (available_cores - 2) cores
sfInit(parallel = TRUE, cpus = available_cores - 2)
sfLibrary(mvtnorm)  # Ensure mvtnorm is loaded on all worker nodes
sfLibrary(EnvStats)  # Ensure EnvStats is loaded on all worker nodes
sfLibrary(expm) # Ensure expm is loaded on all worker nodes

################################################################################
# Parameter Setup and Dependencies
################################################################################

# Exported variables and functions
source("../utils/functions_estimation.R")  # Adjust path as needed
n=1000
m=100
N=n*m
proportion_list=seq(0,1/sqrt(N),length=5)  # Contamination proportion list
scale_list=c(1,10,100)  # Contamination scale list
type_name=c(
  'freq','freq_plugin','freq_ols','freq_fix',
  'width','width_plugin','width_ols','width_fix'
  )  # Coverage and width list for the four methods (1) - (4)

################################################################################
# Wrapper Function for Simulations
# Run distributed CIs for different configurations of contamination
# proportion and scale.
################################################################################

wrapper<- function(seed){
  
  # Seed for reproducibility
  set.seed(seed + 2020)
  
  # Results container
  FResults <- list()
  count <- 1
  
  # Simulation parameters
  d=50
  n_iter=as.integer(log(m)+1)
  beta.true = rep(1,d)
  beta0.true = 0
  tau0=1
  alpha=0.05 # (1-alpha) is the confidence level for CI
  j=1 # Do inference for the first coornidate of the covariate X
  B=500 # Bootstrapping iterations

  ###############################################################################################
  #  Simulate data (X,y) based on different contamination proportions and scales
  ###############################################################################################
  
  # Loop over contamination proportion and scale list
  for (proportion in proportion_list)
  {
    for (scale in scale_list)
    {
      # Contamination proportion setup
      epsilon=proportion
      # Simulate data
      X = rmvnorm(n = N, mean = rep(0, d), sigma = diag(1,d))
      y = beta0.true + X%*%beta.true 
      X = cbind(rep(1,N),X)
      cont_idx=rbinom(N,1,epsilon)  
      # Contamination scale setup (we use F = t(1.5), G = scale*X1 here)
      y=y+(1-cont_idx)*rt(N, 1.5)+cont_idx*(scale*X[,2])
      
      ###############################################################################################
      #  Initializer for distributed Huber estimator: average of m local Huber estimators
      ###############################################################################################

      index = function(c) ((c-1)*n+1):(c*n) # Helper function to decide index ranges for each local sample
      beta_dc = rowMeans(sapply(1:m, function(x) Huber_GDBB(y[index(x)], X[index(x),], tau_const = tau_adapt0(y[index(x)], X[index(x),],tau0))))

      ###############################################################################################
      #  Distributed Huber estimator
      ###############################################################################################

      ### Step 1: Use the local sample data from the first (master) machine to adaptively select tau
      y1=y[index(1)]
      X1=X[index(1),]
      tau_const=tau_adapt0(y1,X1,tau0)

      ### Step 2: Obtain distributed Huber estimator with the above tau and the above initializer
      beta.dis = Distributed_Huber(y, X, m, n.iter = n_iter,intial=T, beta0 = beta_dc, tau_const = tau_const)
      l=length(beta.dis)
      beta_hat=beta.dis[[l-1]] # Distributed Huber estimator: beta^(T), where T = n_iter
      beta0=beta.dis[[l-2]] # Distributed estimator at iteration T-1: beta^(T-1), where T = n_iter
      grad_shift=beta.dis[[l]] # Global gradient at beta^(T)

      ###############################################################################################
      # Method (1) M-boot: distributed bootstrap CI for distributed M estimator
      ###############################################################################################

      ### Distributed bootstrap estimators (B estimators in total)
      beta_boot_full=matrix(0,nrow=B,ncol=ncol(X))
      for (b in 1:B)
      {
        beta_boot_full[b,] = Huber_GDBB_w(y1, X1, tau_const = tau_const, distributed = T, beta0 =beta0 , gradient_shift = grad_shift)
      }

      ### M-boot using Bootstrap quantiles
      CI_1=beta_hat[j+1]-1/sqrt(m)*(quantile(beta_boot_full[,j+1],1-alpha/2)-beta_hat[j+1])
      CI_2=beta_hat[j+1]-1/sqrt(m)*(quantile(beta_boot_full[,j+1],alpha/2)-beta_hat[j+1])
      CI_width=CI_2-CI_1
      CI_freq=(beta.true[j]<CI_2)*(beta.true[j]>CI_1)

      ###############################################################################################
      # Method (2) Debias-M-boot: debiased distributed bootstrap CI for distributed M estimator
      ###############################################################################################

      ### estimates required by debiased CI
      r=y1-X1%*%beta_hat  # residuals
      c_F=mean(abs(r)<=tau_const)  # mean of the second derivative of l(residuals), where l is the Huber loss
      S_1_sq=sqrtm(solve(t(X1)%*%X1/n))  # S^{-1/2} estimate from the local sample in the first (master) machine
      q_fix=epsilon*tau_const/(c_F*sqrt(sum((S_1_sq[j+1,])^2))) # debias quantity (we use C_D = 1 here)

      ### debias CI
      CI_1_fix=CI_1-q_fix
      CI_2_fix=CI_2+q_fix
      CI_width_fix=CI_2_fix-CI_1_fix
      CI_freq_fix=(beta.true[j]<CI_2_fix)*(beta.true[j]>CI_1_fix)

      ###############################################################################################
      # Method (3) M-normal: asymptotic normality based CI for distributed M estimator
      ###############################################################################################

      ### estimates required by M-normal CI
      score=r*(abs(r)<=tau_const)+tau_const*(abs(r)>tau_const)*sign(r)  # the gradient vector of l(residual), where l is the Huber loss
      C_F_sq=sd(score)  # estimate of the square root of C_F
      S_1=solve(t(X1)%*%X1/n)  # estimate of S^{-1} using the local sample from the first (master) machine
      sd_U_plugin=C_F_sq/c_F*sqrt(S_1[j+1,j+1])  # estimate of the SD of the distributed Huber estimator

      ### M-normal CI
      CI_1_norm_plugin=beta_hat[j+1]-1/sqrt(N)*(qnorm(1-alpha/2)*sd_U_plugin)
      CI_2_norm_plugin=beta_hat[j+1]-1/sqrt(N)*(qnorm(alpha/2)*sd_U_plugin)
      CI_width_norm_plugin=CI_2_norm_plugin-CI_1_norm_plugin
      CI_freq_norm_plugin=(beta.true[j]<CI_2_norm_plugin)*(beta.true[j]>CI_1_norm_plugin)

      
      ###############################################################################################
      # Method (4) OLS-normal: asymptotic normality based CI for global OLS estimator
      ###############################################################################################   

      ### Global OLS estimator
      S_1_global=solve(t(X)%*%X/N)
      beta_ols_global=as.vector(S_1_global%*%(t(X)%*%y)/N)
      
      ### Global OLS CI
      r_global=y-X%*%beta_ols_global
      sd_ols_global=sd(r_global)*sqrt(S_1_global[j+1,j+1])
      CI_1_ols=beta_ols_global[j+1]-1/sqrt(N)*(qnorm(1-alpha/2)*sd_ols_global)
      CI_2_ols=beta_ols_global[j+1]-1/sqrt(N)*(qnorm(alpha/2)*sd_ols_global)
      CI_width_ols=CI_2_ols-CI_1_ols
      CI_freq_ols=(beta.true[j]<CI_2_ols)*(beta.true[j]>CI_1_ols)
      
      ###############################################################################################
      #  Collect results
      ############################################################################################### 
      
      FResults0=c(CI_freq,CI_freq_norm_plugin,CI_freq_ols,CI_freq_fix,
                  CI_width,CI_width_norm_plugin,CI_width_ols,CI_width_fix)      
      FResults[[count]]=FResults0
      count=count+1
    }
  }
  return (FResults)
} 


#################################################################################
#  Repeatedly run the wrapper function 500 times:
#################################################################################
nSim <- 500
#################################################################################
#  Exporting the data and functions needed by the worker nodes:
#################################################################################
sfExport("N", "n", "m", "proportion_list", "scale_list", 
         "grad", "tau_adapt0", "Huber_GDBB", "Huber_GDBB_w", "Distributed_Huber", "wrapper")
FinalResult <- sfLapply(1:nSim,wrapper)


### Write the final results into a table
iteration=rep(c(1:nSim),each=length(proportion_list)*length(scale_list)*length(type_name))
proportion=rep(rep(proportion_list,each=length(scale_list)*length(type_name)),nSim)
scale=rep(rep(scale_list,each=length(type_name)),length(proportion_list)*nSim)
type=rep(type_name,nSim*length(proportion_list)*length(scale_list))
results=unlist(FinalResult)
df=data.frame(iteration,proportion,scale,type,results)

### save table
write.csv(df, file = "results/output_CI_epsilon.csv")
sfStop()
