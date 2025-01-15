################################################################################
# Helper Functions for Distributed Robust Estimation and Bootstrap Inference
################################################################################

# Gradient function for Huber, Pseudo-Huber, and smooth-Huber loss functions
# (See Appendix A of our paper for details).
# Input:
#   y: Response vector
#   X: Design matrix
#   beta: Current parameter estimates
#   tau: Tuning parameter for robust loss
#   method: Loss function to use ("Huber", "Pseudo-Huber I", etc.)
#   distributed: Boolean; TRUE if distributed gradient is used
#   gradient_shift: Gradient shift for distributed settings (if applicable)
#   weight: Optional weight vector for bootstrap or reweighted estimators
# Output:
#   Gradient vector for the specified loss function
grad = function(y, X, beta, tau, method = "Huber", distributed = F, gradient_shift = NULL, weight = 1){
  
  n = nrow(X)
  r=y-X%*%beta # Residuals
  
  # Compute score based on the chosen loss function
  if (method=="Huber")
  {
    score=r*(abs(r)<=tau)+(tau)*(abs(r)>tau)*sign(r)
  }
  else if (method=="Pseudo-Huber I")
  {
    score=tau*r/sqrt(tau^2+r^2)
  }
  else if (method=="Pseudo-Huber II")
  { 
    rt=r/tau
    score=tau*(exp(rt-abs(rt))-exp(-rt-abs(rt)))/(exp(rt-abs(rt))+exp(-rt-abs(rt)))
  }
  else if (method=="Smooth-Huber I")
  {
    rt=r/tau
    score=tau*((rt-sign(rt)*rt^2/2)*(abs(rt)<=1)+sign(rt)/2*(abs(rt)>1))
  }
  else if (method=="Smooth-Huber II")
  {
    rt=r/tau
    score=tau*((rt-rt^3/6)*(abs(rt)<=sqrt(2))+2*sqrt(2)/3*sign(rt)*(abs(rt)>sqrt(2)))
  }
  
  # Apply weights to the score
  score = weight * score
  summand=as.vector(-t(X)%*%score)/n
  
  # Adjust for distributed gradient if needed, for which a gradient_shift (i.e., local grad - global grad) is applied
  if (distributed) return(summand - gradient_shift)
  else return(summand)
}


# Adaptive selection of tau for the Huber estimator
# Input:
#   y: Response vector
#   X: Design matrix
#   tau0: Initial value of tau
#   method: Loss function to use ("Huber", "Pseudo-Huber I", etc.)
# Output:
#   Adaptively selected tau (Median Absolute Deviation of residuals)
tau_adapt0 = function(y, X, tau0, method = "Huber"){
  # Step 1: obtain a Huber estimator and its residuals using an initial value of tau0
  beta0=Huber_GDBB(y, X, tau_const = tau0, method = method)
  r = y-X%*%beta0
  # Step 2: set tau = MAD(residuals)
  med = median(r)
  return(median(abs(r - med)))
}

################################################################################
# Core Algorithms for Robust Estimation
################################################################################

# A gradient descent (GD) algorithm for Huber loss (and other Pseudo-Huber/smooth-Huber loss) 
# with Barzilai-Borwein (BB) stepsize selection
# Input:
#   y, X: Response vector and design matrix
#   tau_const: Tuning parameter for Huber loss
#   method: Loss function to use ("Huber", "Pseudo-Huber I", etc.)
#   distributed: Boolean; TRUE for distributed settings
#   beta0: Initial parameter estimates
#   gradient_shift: Gradient shift for distributed settings
#   delta: Convergence threshold for gradient norm
#   max_iter: Maximum number of iterations
# Output:
#   Final parameter estimates after gradient descent
Huber_GDBB <- function(y, X, tau_const, method = "Huber", distributed = F, beta0 = rep(0, ncol(X)), gradient_shift = NULL, delta = 1e-5, max_iter = 1000){
  
  n = nrow(X)
  # Algorithm loop starts
  if (distributed == F){
    beta.t_1 = beta0 = rep(0, ncol(X)) # Initializer is set as 0 in a non-distributed setting
  }else{
    beta.t_1 = beta0 # In a distributed setting, an initializer is supposed to be given
  }
  
  grad.t_1 = grad.0 = grad(y, X, beta.t_1, tau_const, method = method, distributed = distributed, gradient_shift = gradient_shift)
  beta.t = beta.t_1 - grad.t_1
  grad.t = grad(y, X, beta.t, tau_const, method = method, distributed = distributed, gradient_shift = gradient_shift)
  count = 0
  
  # Iteratively do gradient descent until the l-2 norm of the gradient is less than a given threshold delta or reach max_iter
  for (count in 1:max_iter)
  { if (sqrt(sum(grad.t^2)) < delta) break
    else
    {
      # Calculate stepsize using BB method
      delta.t = beta.t - beta.t_1
      g.t = grad.t - grad.t_1
      eta.1t = as.vector((delta.t %*% delta.t+1e-7)/ (delta.t %*% g.t+1e-7))
      eta.2t = as.vector((delta.t %*% g.t+1e-7)/ (g.t %*% g.t+1e-7))
      eta.t = ifelse(min(eta.1t,eta.2t) > 0, min(eta.1t, eta.2t, 1), 1)
      # Gradient descent and update parameters
      beta.t_1 = beta.t
      beta.t = beta.t_1 - eta.t*grad.t
      grad.t_1=grad.t
      grad.t = grad(y, X, beta.t, tau_const, method = method, distributed = distributed, gradient_shift = gradient_shift)
    }

  }
  return(beta.t) 
}



# Distributed Huber Estimator Algorithm
# Input:
#   y, X: Response vector and design matrix
#   m: Number of machines 
#   n.iter: Number of surrogate loss minimization steps
#   intial: Boolean; If TRUE, use the provided initializer (beta0). 
#           If FALSE (default), use the local estimator from the first 
#           (master) machine as the initializer.
#   beta0: Initial parameter estimates
#   tau_const: Tuning parameter for Huber loss
#   method: Loss function to use ("Huber", "Pseudo-Huber I", etc.)
# Output:
#   List of parameter estimates at each step and the final gradient shift
Distributed_Huber <- function(y, X, m, n.iter = 2, intial = F, beta0 = rep(0, ncol(X)), tau_const, method = "Huber"){
  N = length(y)
  n = N/m
  # Helper function to determine index ranges for each local sub-sample
  index = function(c) ((c-1)*n+1):(c*n)
  
  # If intial==F, then initialize beta^(0) as the local estimator using the local sample from first (master) machine
  if(intial==F) 
  {beta0=Huber_GDBB(y[index(1)], X[index(1),], tau_const = tau_const, method = method)}
  beta.list=list(beta0)
  # Local grad for the first (master) machine
  grad0_1=grad(y[index(1)], X[index(1),], beta0, tau_const, method = method)
  # Global grad, i.e., the average of the local grads over m machines
  grad0_N = rowMeans(sapply(1:m, function(x) grad(y[index(x)], X[index(x),], beta0, tau_const, method = method)))
  
  # Solve surrogate loss minimization using Huber_GDBB, needs input of constant gradient shift and first machine's data.
  for (tt in 1:n.iter)
  {  
    grad_shift = grad0_1-grad0_N
    beta= Huber_GDBB(y[index(1)], X[index(1),], tau_const = tau_const, method = method, distributed = T, beta0 = beta0, gradient_shift = grad_shift)
    # Update local grad and global grad with the updated estimator beta
    grad_1 = grad(y[index(1)], X[index(1),], beta, tau_const) 
    grad_N = rowMeans(sapply(1:m, function(x) grad(y[index(x)], X[index(x),], beta, tau_const)))
    # Stop if global grad has a larger l-infty norm
    if ((max(abs(grad_N))>max(abs(grad0_N)))&(tt>=2)) break
    # Else, save the estimator beta at this iteration and update parameters
    else
    {
      beta.list[[tt+1]] = beta
      beta0=beta
      grad0_1=grad_1
      grad0_N=grad_N
    }

  }
  return (c(beta.list,list(grad_shift))) # We save grad_shift as it would be used for distributed inference
}


################################################################################
# Core Algorithms for Bootstrap Inference
################################################################################

# Distributed Multiplier Bootstrap Estimator
# Description:
#   Uses weights \( w_i \) generated from a scaled Bernoulli distribution 
#   with \( p = 0.5 \).
# Input:
#   y, X: Response vector and design matrix
#   Other arguments: Same as in `Huber_GDBB`, including:
#     - tau_const: Tuning parameter for Huber loss
#     - method: Loss function to use ("Huber", "Pseudo-Huber I", etc.)
#     - distributed: Boolean; TRUE for distributed settings
#     - beta0: Initial parameter estimates
#     - gradient_shift: Gradient shift for distributed settings
#     - delta: Convergence threshold for gradient norm
#     - max_iter: Maximum number of iterations
# Output:
#   Final bootstrap estimator

Huber_GDBB_w <- function(y, X, tau_const, method = "Huber", distributed = F, beta0 = rep(0, ncol(X)), gradient_shift = NULL, delta = 1e-5, max_iter = 1000){
  n = nrow(X)
  # Weight w_i generation 
  weight = 2*rbinom(n, 1, 0.5)
  
  # Algorithm loop starts
  if (distributed == F){
    beta.t_1 = beta0 = rep(0, ncol(X)) 
  }else{
    beta.t_1 = beta0
  }
  
  grad.t_1 = grad.0 = grad(y, X, beta.t_1, tau_const, method = method, distributed = distributed, gradient_shift = gradient_shift, weight = weight)
  beta.t = beta.t_1 - grad.t_1
  grad.t = grad(y, X, beta.t, tau_const, method = method, distributed = distributed, gradient_shift = gradient_shift, weight = weight)
  
  # Iteratively do gradient descent until the l-2 norm of the gradient is less than a given threshold delta or reach max_iter
  count = 0
  while ((sqrt(sum(grad.t^2)) > delta) & (count <= max_iter)){
    # Calculate stepsize using BB method
    delta.t = beta.t - beta.t_1
    g.t = grad.t - grad.t_1
    eta.1t = as.vector((delta.t %*% delta.t+1e-7)/ (delta.t %*% g.t+1e-7))
    eta.2t = as.vector((delta.t %*% g.t+1e-7)/ (g.t %*% g.t+1e-7))
    eta.t = ifelse(min(eta.1t,eta.2t) > 0, min(eta.1t, eta.2t, 1), 1)
    # Gradient descent and update parameters
    beta.t_1 = beta.t
    beta.t = beta.t_1 - eta.t*grad.t
    grad.t_1=grad.t
    grad.t = grad(y, X, beta.t, tau_const, method = method, distributed = distributed, gradient_shift = gradient_shift, weight = weight)
    
    count = count + 1
  }
    return(beta.t) 
}

