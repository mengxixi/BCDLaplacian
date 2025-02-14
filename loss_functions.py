import numpy as np
from scipy.special import logsumexp
from scipy.linalg import eigh

from numba import jit

def create_lossObject(loss_name, A, b, args):
    if loss_name == "ls":
        return Least_Square(A, b, args)

    if loss_name == "lg":
        return Logistic(A, b, args)

    if loss_name == "sf":
        return Softmax(A, b, args)

    if loss_name == "bp":
        return BeliefPropagation(A, b, args)

    if loss_name == "bp_huber":
        return BeliefPropagation_Huber(A, b, args)

    if loss_name == "bp_lg":
        return BeliefPropagation_Logistic(A, b, args)

    if loss_name == "lsl1nn":
        return Least_Square_L1_NN(A, b, args)

        
###########################
# 2. Least_Square
###########################
class Least_Square:
  def __init__(self, A, b, args):
    self.ylabel = "bp loss: $f(x) = \\frac{1}{2} x^T A x - b^Tx$"
    self.L2 = args["L2"]
    self.n_params = A.shape[1]
    self.lipschitz = np.sum(A ** 2, axis=0) + self.L2

  def f_func(self, x, A, b):
    reg = 0.5 * self.L2 * np.sum(x ** 2) 

    b_pred = np.dot(A, x) 
    
    loss = 0.5 * np.sum((b_pred - b)**2) + reg

    return loss

  def g_func(self, x, A, b, block=None):
    b_pred = np.dot(A, x)
    residual = b_pred - b
    if block is None:
      grad = np.dot(A.T, residual)

      grad += self.L2 * x
    else:
      grad = A[:, block].T.dot(residual)
      grad += (self.L2 *  x[block])

    return grad 

  def h_func(self, x, A, b, block=None):
    #b_pred = np.dot(A, x)

    if block is None:
      hessian = np.dot(A.T, A)
      hessian += self.L2 * np.identity(self.n_params)
      
    elif block.size == 1:
      #import ipdb; ipdb.set_trace()
      hessian = np.sum(A[:, block[0]]**2)      
      hessian += self.L2
      
    else:
      # Block case
      hessian = np.dot(A[:, block].T, A[:, block])
      hessian += self.L2 * np.identity(block.size)

    return hessian 

  def Lb_func(self, x, A, b, block=None):
    if block is None:
      E = np.linalg.eig(A.T.dot(A))[0]
      L_block = np.max(E) + self.L2
    else:
      A_b = A[:, block]

      E = np.linalg.eig(A_b.T.dot(A_b))[0]
      L_block = np.max(E) + self.L2
    
    return L_block

  def Hb_func(self, x, A, b, block=None):
    if block is None:
      L_block = np.dot(A.T, A)
      L_block += self.L2 * np.identity(self.n_params)
    else:
      A_b = A[:, block]
      L_block = A_b.T.dot(A_b) + self.L2 * np.identity(block.size)
    
    return L_block


###########################
# 2. Least_Square_L1_NN
###########################
class Least_Square_L1_NN:
  def __init__(self, A, b, args):
    self.ylabel = "bp loss: $f(x) = \\frac{1}{2} x^T A x - b^Tx$"
    self.L2 = args["L2"]
    self.n_params = A.shape[1]
    self.L1 = args["L1"]
    
    assert self.L1 != 0

    self.lipschitz = np.sum(A ** 2, axis=0) + self.L2

  def f_func(self, x, A, b, assert_nn=1):
    # Least squares function    
    assert np.all(x >= 0)
    
    b_pred = np.dot(A, x) 
    
    loss = 0.5 * np.sum((b_pred - b)**2) + self.L1 * np.sum(x)

    return loss

  def g_func(self, x, A, b, block=None):
    # Least squares function
    b_pred = np.dot(A, x)
    residual = b_pred - b
    
    if block is None:
      grad = np.dot(A.T, residual)

      grad += self.L2 * x
      grad += self.L1

    else:
      grad = A[:, block].T.dot(residual)
      grad += (self.L2 *  x[block])
      grad += self.L1
    
    return grad 

  def h_func(self, x, A, b, block=None):
    # Least squares function
    #b_pred = np.dot(A, x)

    if block is None:
      hessian = np.dot(A.T, A)
      hessian += self.L2 * np.identity(self.n_params)
      
    elif block.size == 1:
      #import ipdb; ipdb.set_trace()
      hessian = np.sum(A[:, block[0]]**2)      
      hessian += self.L2
      
    else:
      # Block case
      hessian = np.dot(A[:, block].T, A[:, block])
      hessian += self.L2 * np.identity(block.size)

    return hessian 

  def Lb_func(self, x, A, b, block=None):
    # Least squares function
    if block is None:
      E = np.linalg.eig(A.T.dot(A))[0]
      L_block = np.max(E) + self.L2
    else:
      A_b = A[:, block]

      E = np.linalg.eig(A_b.T.dot(A_b))[0]
      L_block = np.max(E) + self.L2
    
    return L_block

  def Hb_func(self, x, A, b, block=None):
    if block is None:
      L_block = np.dot(A.T, A)
      L_block += self.L2 * np.identity(self.n_params)
    else:
      A_b = A[:, block]
      L_block = A_b.T.dot(A_b) + self.L2 * np.identity(block.size)
    
    return L_block

###########################
# 2. LOGISTIC
###########################
class Logistic:
  def __init__(self, A, b, args):
    self.label = "Binary logistic loss"
    self.L2 = args["L2"]
    self.n_params = A.shape[1]

    self.lipschitz = 0.25 * np.sum(A ** 2, axis=0) + self.L2

    # Mipschitz

    constant = (1. / (6. * np.sqrt(3))) 

    # correct 
    self.mipschitz  = np.sum(np.abs(b**3 * (A ** 3).T)  * constant, axis=1)



  def f_func(self, x, A, b):
    # Logistic function
    reg = 0.5 * self.L2 * np.sum(x ** 2) 

    b_pred = np.dot(A, x) 
    agree = - b*b_pred
    zeros = np.zeros(A.shape[0])

    loss = logsumexp(np.stack([zeros, agree], axis=1), axis=1)
    #loss2 = np.sum(np.log(1 + np.exp(- b*b_pred))) + reg

    return loss.sum() + reg

  def g_func(self, x, A, b, block=None):
    # Logistic function
    b_pred = np.dot(A, x)
    residual = - b / (1 + np.exp(b * b_pred))
    
    if block is None:
      grad = np.dot(A.T, residual)

      grad +=self.L2 * x
    else:
      grad = A[:, block].T.dot(residual)
      grad += (self.L2 *  x[block])

    return grad 

  def h_func(self, x, A, b, block=None):
    # Logistic function
    b_pred = np.dot(A, x)

    sig = 1. / (1. + np.exp(- b * b_pred))

    if block is None:
      S = np.diag(sig * (1-sig))
      hessian = np.dot(A.T, S).dot(A)
      hessian += self.L2 * np.identity(x.size)
      
    elif block.size == 1:
      #import ipdb; ipdb.set_trace()
      hessian = np.sum(A[:, block[0]]**2 * sig * (1-sig))      
      hessian += self.L2
      
    else:
      # Block case
      p = np.diag(sig * (1- sig))
      hessian = A[:, block].T.dot(p).dot(A[:, block])
      hessian +=self.L2 * np.identity(block.size)

    return hessian 

  def Lb_func(self, x, A, b, block=None):
    # Logistic function
    if block is None:
      A_b = A 
    else:
      A_b = A[:, block]

    E = np.linalg.eig(A_b.T.dot(A_b))[0]
    L_block = 0.25 * np.max(E) + self.L2
    
    return L_block


  def Hb_func(self, x, A, b, block=None):
    # Logistic function
    if block is None:
      A_b = A

      L_block = 0.25 * A_b.T.dot(A_b) + self.L2
    
    else:
      A_b = A[:, block]

      L_block = 0.25 * A_b.T.dot(A_b) + self.L2
    
    return L_block

###########################
# 2. Softmax
###########################

class Softmax:
  def __init__(self, A, b, args):
    self.label = "Multiclass logistic loss"
    self.L2 = args["L2"]
    self.n_params = A.shape[1] * b.shape[1]
    self.n_features = A.shape[1]
    self.n_classes = b.shape[1]

    self.lipschitz = (np.ones((self.n_classes, 1)) * 0.25 * 
                      np.sum(A ** 2, axis=0) + self.L2).T
    self.lipschitz = self.lipschitz.ravel()

    # Mipschitz
    sigma = (np.sqrt(3.) + 3.)/6.
    constant = (1 - sigma) * sigma * (1 - 2 * sigma)
      

    self.mipschitz =  (np.ones((self.n_classes, 1)) * 
                        np.sum(np.abs(constant * A ** 3), axis=0)).T

    self.mipschitz = self.mipschitz.ravel()



  def base(self, x, block):
    # Softmax function
    if block is not None and not isinstance(block, tuple):
      block = np.unravel_index(block,  (self.n_features, self.n_classes))

    # Reshape x
    if x.ndim == 2:
      assert x.shape[0] == self.n_features
      assert x.shape[1] == self.n_classes
    else:
      assert x.ndim == 1
      x = x.reshape((self.n_features, self.n_classes))

    return x, block

  def f_func(self, x, A, b):
    # Softmax function
    x, block = self.base(x, block=None)

    b_pred = np.dot(A, x)
    # loss = - np.sum(b * np.log(softmax(b_pred)))

    # Add normalizing factors
    loss = np.sum(logsumexp(b_pred, axis=1))
    # Add dot products
    loss -= np.sum(b_pred * b)
    # Add regualization
    reg = 0.5 * self.L2 * np.sum(x ** 2)
    loss += reg

    return loss

  def g_func(self, x, A, b, block=None):
    # Softmax function
    x, block = self.base(x, block)
    b_pred = np.dot(A, x)
    R = softmax(b_pred) - b

    if block is None:
        grad = np.dot(A.T, R)
        grad += self.L2 * x

    else:
        features, classes = block
        # GET THE DIAGONAL OF THE DOT PRODUCT
        grad = np.einsum('ij,ij->i', A.T[features], R[:, classes].T)
        grad += self.L2 * x[block]

        # features, classes = block

        # grad = np.dot(A[:, features].T, 
        #               softmax(b_pred)[:, classes] - b[:, classes])

        # grad += (L2 *  x[features, classes])

    return grad.ravel()

  def h_func(self, x, A, b, block=None):
    # Softmax function
    x, block = self.base(x, block)
    b_pred = np.dot(A, x)

    if block == None:
      qweqwe
      block = np.arange(x.size)
      block = np.unravel_index(block,  (self.n_features, self.n_classes))

    if block[0].size == 1:
        # One coordinate
        features, classes = block
        #import pdb;pdb.set_trace()
        soft = softmax(b_pred)[:, classes]

        h = np.sum(A[:, features]**2 * soft * (1-soft))
        h += self.L2

    else:

      # Block Coordinate
      features, classes = block
      soft = softmax(b_pred)

      block_size = features.size

      h = np.zeros((block_size, block_size))
      for i, (f1, c1) in enumerate(zip(features, classes)):
        for j, (f2, c2) in enumerate(zip(features, classes)):
          # Class sigmoid 
          if c1 == c2:
            sig = soft[:, c1]
            S = sig * (1-sig)
          else:
            S = -soft[:, c1] * soft[:,c2]

          # Feature
          h[i, j] = np.dot(S, A[:, f1]* A[:, f2])

      h += self.L2 * np.identity(block_size)  

    return h

  
  def Lb_func(self, x, A, b, block=None):
    # Softmax function
    x, block = self.base(x, block)
    Hb = self.Hb_func(x, A, b, block=block)    
    
    E = eigh(Hb, eigvals_only=True)
    L_block = np.max(E) + self.L2
    
    return L_block


  def Hb_func(self, x, A, b, block=None):
    # Softmax function
    x, block = self.base(x, block)

    if block is None:
      features, classes = np.mgrid[:self.n_features, :self.n_classes]
      features = features.ravel()
      classes = classes.ravel()
    else:
      features, classes = block
    n_classes = b.shape[1]

    block_size = features.size

    if block is None:
      n, d = A.shape
      k = b.shape[1]
      
      Hb = computeFull_Hb(A, d, k, features, classes)
    else:
      U = np.unique(features)
      Umap = {u:i for i, u in enumerate(U)}
      k = min(n_classes, np.unique(classes).size + 1)
          
      Z = A[:, U]
      Z = Z.T.dot(Z)


      Hb = np.zeros((block_size, block_size))

      for i, (f1, c1) in enumerate(zip(features, classes)):
        for j, (f2, c2) in enumerate(zip(features, classes)):

          # Class sigmoid 
          if c1 == c2:
            s = 0.5 * (1. - 1. / k)
          else:
            s = 0.5 * (0. - 1. / k)

          # Feature
          #Hb[i, j] = np.sum(s * A[:, f1] * A[:, f2])
          Hb[i, j] = s * Z[Umap[f1], Umap[f2]]
    if self.L2:
      Hb += self.L2 * np.identity(block_size)     
    
    return Hb

def softmax(X):
    #import ipdb; ipdb.set_trace()
    tmp = X - X.max(axis=1)[:, np.newaxis]
    X = np.exp(tmp)

    X /= X.sum(axis=1)[:, np.newaxis]

    return X

@jit
def computeFull_Hb(A, d, k, features, classes):
  AA = A.T.dot(A)
  Hb = np.zeros((d*k, d*k))

  for i in range(d*k):
    f1 = features[i]
    c1 = classes[i]
    for j in range(d*k):
      f2 = features[j]
      c2 = classes[j]

      # Class sigmoid 
      if c1 == c2:
        s = 0.5 * (1. - 1. / k)
      else:
        s = 0.5 * (0. - 1. / k)

      Hb[i, j] = s * AA[f1, f2]

  return Hb


###########################
# 2. Belief Propagation
###########################
class BeliefPropagation:
  def __init__(self, A, b, args):
    self.ylabel = "bp loss: $f(x) = \\frac{1}{2} x^T A x - b^Tx$"
    self.L2 = args["L2"]
    self.n_params = A.shape[1]

    self.lipschitz = np.diag(A) + self.L2

  def f_func(self, x, A, b):
    # BeliefPropagation
    reg = 0.5 * self.L2 * np.sum(x ** 2)
      
    loss = 0.5 * x.T.dot(A).dot(x)
    loss -= np.dot(b, x)
    loss += reg

    return loss

  def g_func(self, x, A, b, block=None):
    # BeliefPropagation
    L2 = self.L2
    
    if block is None:
      grad = np.dot(x.T, A) - b
      grad += L2 * x
      
    else:
      all_indices = np.arange(self.n_params)
    
      block_indices = block
      non_block_indices = np.delete(all_indices, block_indices)
    
      A_bc = A[block_indices][:, non_block_indices]
      A_bb = A[block_indices][:, block_indices]
      
      y_prime = A_bc.dot(x[non_block_indices]) - b[block_indices]
      X_prime = np.dot(A_bb, x[block_indices])
      grad = X_prime + y_prime
      
      
      grad += (L2 *  x[block])

    return grad

  def h_func(self, x, A, b, block=None):
    # BeliefPropagation
    """dual hess"""
    if block is None:
      return A + self.L2 * np.identity(self.n_params)
    else:
      return A[block][:, block] + self.L2 * np.identity(block.size)


  def Lb_func(self, x, A, b, block=None):
    # BeliefPropagation
    if block is None:
      A_b = A
    else:
      A_b = A[block][:, block]

    E = np.linalg.eigh(A_b)[0]
    L_block = np.max(E) + self.L2
    
    return L_block

  def Hb_func(self, x, A, b, block=None):
    # BeliefPropagation

    if block is None:
      A_b = A
    else:
      A_b = A[block][:, block]

    L_block = A_b + self.L2
    
    return L_block


###########################
# 3. Belief Propagation with Huber loss
###########################
class BeliefPropagation_Huber:
  def __init__(self, A, b, args):
    self.ylabel = "bp huber"
    self.n_params = A.shape[1]
    self.W = args["data_W"]
    self.y = args["data_y"]
    self.unlabeled_indices = args["unlabeled"]
    self.labeled_indices = args["labeled"]
    self.eps = 1.0

    self.L2 = args["L2"]
    self.n_params = A.shape[1]
    self.lipschitz = np.diag(A) + self.L2

  
  def huber(self, z):
    return np.where(np.abs(z)<=self.eps, 0.5*z**2, self.eps*(np.abs(z)-0.5*self.eps))


  def huber_p(self, z):
    return np.where(np.abs(z)<=self.eps, z, self.eps*np.sign(z))


  def huber_pp(self, z):
    return np.where(np.abs(z)<=self.eps, 1, 0)


  def pairwise_differences(self, x, y):
    return np.expand_dims(x,1)-y


  def f_func(self, ybar, A=None, b=None):
    # Unlabeled
    W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
    z = self.pairwise_differences(ybar, ybar)
    z = W_UU * self.huber(z)
    loss = np.sum(z)

    # Labeled
    W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
    ylabled = self.y[self.labeled_indices]
    z = self.pairwise_differences(ybar, ylabled)
    z = W_UL * self.huber(z)
    loss += np.sum(z)

    # Regularization
    loss += 0.5 * self.L2 * np.sum(ybar**2)

    return loss

  def g_func(self, ybar, A=None, b=None, block=None):
    if block is None:
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      z = self.pairwise_differences(ybar, ybar)
      grad = 2*np.sum(W_UU*self.huber_p(z), axis=1)
      grad -= 3*np.diag(W_UU)*self.huber_p(np.diag(z))

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_differences(ybar, ylabled)
      grad += np.sum(W_UL*self.huber_p(z), axis=1)

      # Regularization
      grad += (self.L2*ybar)

    else:    
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      W_UB = W_UU[block, :]
      z = self.pairwise_differences(ybar[block], ybar)
      grad = 2*np.sum(W_UB*self.huber_p(z), axis=1)
      grad -= 3*np.diag(W_UB)*self.huber_p(np.diag(z))

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      W_BL = W_UL[block, :]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_differences(ybar[block], ylabled)
      grad += np.sum(W_BL*self.huber_p(z), axis=1)

      # Regularization
      grad += (self.L2 * ybar[block])

    return grad

  def h_func(self, ybar, A, b, block=None):
    if block is None:
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      z = self.pairwise_differences(ybar, ybar)
      hpp = self.huber_pp(z)
      Z = W_UU*hpp      
      h = -2*Z + np.diag(2*np.sum(Z, axis=1)+np.diag(Z))

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_differences(ybar, ylabled)
      hpp = self.huber_pp(z)
      h += np.diag(np.sum(W_UL*hpp, axis=1))

      # Regularization
      h += self.L2 * np.identity(len(ybar))

    else:
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      W_UB = W_UU[block, :]
      z = self.pairwise_differences(ybar[block], ybar)
      Z = W_UB[:,block]*self.huber_pp(z[:,block])
      h = -2*Z + np.diag(2*np.sum(W_UB*self.huber_pp(z), axis=1)+np.diag(Z))

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      W_BL = W_UL[block,:]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_differences(ybar[block], ylabled)
      hpp = self.huber_pp(z)
      h += np.diag(np.sum(W_BL*hpp, axis=1))

      # Regularization
      h += self.L2 * np.identity(len(block))

    return h

  def Lb_func(self, x, A, b, block=None):
    # BeliefPropagation
    if block is None:
      A_b = A
    else:
      A_b = A[block][:, block]

    E = np.linalg.eigh(A_b)[0]
    L_block = np.max(E) + self.L2
    
    return L_block

  def Hb_func(self, x, A, b, block=None):
    # BeliefPropagation
    return self.h_func(x, A, b, block)


###########################
# 4. Belief Propagation with Logistic loss
###########################
class BeliefPropagation_Logistic:
  def __init__(self, A, b, args):
    self.ylabel = "bp logistic"
    self.n_params = A.shape[1]
    self.W = args["data_W"]
    self.y = args["data_y"]
    self.unlabeled_indices = args["unlabeled"]
    self.labeled_indices = args["labeled"]

    # Need some form of regularization otherwise it'll go to infty
    self.L2 = 1e-3 if args["L2"] == 0 else args["L2"]
    self.n_params = A.shape[1]

    self.lipschitz = np.diag(A) + self.L2

  
  def logistic(self, z):
    zeros = np.zeros(z.shape)
    return logsumexp(np.stack([zeros, z], axis=2), axis=2)


  def logistic_p(self, z):
    return 1/(1+np.exp(-z))-1


  def logistic_pp(self, z):
    expz = np.exp(z)
    return expz/((1+expz)**2)


  def pairwise_products(self, x, y):
    return np.expand_dims(x,1)*y


  def f_func(self, ybar, A=None, b=None):
    # Unlabeled
    W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
    z = self.pairwise_products(ybar, ybar)
    z = W_UU * self.logistic(z)
    loss = np.sum(z)
    
    # Labeled
    W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
    ylabled = self.y[self.labeled_indices]
    z = self.pairwise_products(ybar, ylabled)
    z = W_UL * self.logistic(z)
    loss += np.sum(z)

    # Regularization
    loss += 0.5 * self.L2 * np.sum(ybar**2)

    return loss

  def g_func(self, ybar, A=None, b=None, block=None):
    if block is None:
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      z = self.pairwise_products(ybar, ybar)
      g = 2*np.sum(W_UU*self.logistic_p(z)*ybar, axis=1)
      
      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_products(ybar, ylabled)
      g += np.sum(W_UL*self.logistic_p(z)*ylabled, axis=1)

      # Regularization
      g += self.L2*ybar

    else:    
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      W_BU = W_UU[block, :]
      ybar_b = ybar[block]
      z = self.pairwise_products(ybar_b, ybar)
      g = 2*np.sum(W_BU*self.logistic_p(z)*ybar, axis=1)

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      W_BL = W_UL[block, :]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_products(ybar[block], ylabled)
      g += np.sum(W_BL*self.logistic_p(z)*ylabled, axis=1)

      # Regularization
      g += (self.L2 * ybar[block])

    return g

  def h_func(self, ybar, A, b, block=None):
    if block is None:
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      z = self.pairwise_products(ybar, ybar)
      p = self.logistic_p(z)
      pp = self.logistic_pp(z)
      ybarsq = ybar**2
      # Off-diagonals
      h = 2*W_UU*(pp*z+p)
      # Diagonals
      h += np.diag(np.sum(2*W_UU*pp*ybarsq, axis=1) + np.diag(2*W_UU*pp*ybarsq+W_UU*p))

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_products(ybar, ylabled)
      h += np.diag(np.sum(W_UL*self.logistic_pp(z)*(ylabled**2), axis=1))

      # Regularization
      h += self.L2 * np.identity(len(ybar))

    else:
      # Unlabeled
      W_UU = self.W[self.unlabeled_indices][:,self.unlabeled_indices]
      W_BB = W_UU[block][:,block]
      ybar_b = ybar[block]
      z_BU = self.pairwise_products(ybar_b, ybar)
      z = z_BU[:,block]
      p = self.logistic_p(z)
      pp = self.logistic_pp(z)
      ybarsq = ybar_b**2
      # Off-diagonals
      h = 2*W_BB*(pp*z+p)
      # Diagonals
      W_BU = W_UU[block,:]
      h += np.diag(np.sum(2*W_BU*self.logistic_pp(z_BU)*(ybar**2), axis=1) + np.diag(2*W_BB*pp*ybarsq+W_BB*p))

      # Labeled
      W_UL = self.W[self.unlabeled_indices][:,self.labeled_indices]
      W_BL = W_UL[block, :]
      ylabled = self.y[self.labeled_indices]
      z = self.pairwise_products(ybar[block], ylabled)
      h += np.diag(np.sum(W_BL*self.logistic_pp(z)*(ylabled**2), axis=1))

      # Regularization
      h += self.L2 * np.identity(len(block))

    return h

  def Lb_func(self, x, A, b, block=None):
    # BeliefPropagation
    if block is None:
      A_b = A
    else:
      A_b = A[block][:, block]

    E = np.linalg.eigh(A_b)[0]
    L_block = np.max(E) + self.L2
    
    return L_block

  def Hb_func(self, x, A, b, block=None):
    # BeliefPropagation
    return self.h_func(x, A, b, block)