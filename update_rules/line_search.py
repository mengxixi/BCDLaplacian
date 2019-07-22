import numpy as np 
import scipy.linalg as sp

def step(x, d, block, proj):
  xc = x.copy()
  xc[block] = proj(x[block] + d)
  return xc


def perform_line_search(x_old, grad, block, 
                        F, D, 
                        alpha0=1.0, 
                        proj=None):
    x = x_old.copy()
    g = grad
    

    proj = proj or (lambda x: x)

    
    Z = lambda x, phi, alpha: F(step(x, D(alpha), block, proj)) - F(x) - alpha * phi

    t = 1
    eps = 1e-10
    alpha = alpha0
    d = D(alpha)
    phi = np.dot(g, d)

    # Note: technically we should assert <0 since it needs to be a descent 
    # direction, but for label prop initially the unlabeled points are 0
    # and the gradients could be zero if we initially sample a block that 
    # doesn't connect to any of the labeled points.
    assert phi <= 0.0 

    while F(step(x, d, block, proj)) > (F(x) + eps * alpha * phi):

        alphaTmp = alpha
        if t == 1:
          # Quadratic interpolation
          z = Z(x, phi, alpha)
          alpha =  - (phi*alpha**2) / (2 * z)

          # Keep track
          zOld = z
          alphaOld = alphaTmp

        elif t > 1:
          # Cubic interpolation
          c = 1. / ((alpha - alphaOld) * (alphaOld**2) * alpha**2 )
          z = Z(x, phi, alpha)

          a = c * ((alphaOld**2) * z - (alpha**2) * zOld)
          b = c * ((-alphaOld**3) * z + (alpha**3) * zOld)
         
          alpha = (-b + np.sqrt(b**2 - 3 * a * phi)) / (3 * a)

          # Keep track
          zOld = z
          alphaOld = alphaTmp

        # Adjust if change in alpha is too small/large
        if alpha < alphaTmp * 1e-3:
            print("Interpolated value too small, Adjusting")
            alpha = alphaTmp * 1e-3
        elif alpha > alphaTmp * 0.6:
            print("Interpolated value too large, Adjusting")
            alpha = alphaTmp * 0.6

        # Update direction
        d = D(alpha)

        # Check whether step size has become too small
        if np.max(np.abs(alpha*d)) <= 1e-9:
            print("Backtracking Line Search Failed")
            alpha = 0.0
            break

        t += 1

    return alpha

