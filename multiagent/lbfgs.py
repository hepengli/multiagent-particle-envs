import numpy as np
from collections import deque
def lbfgs(f_Ax, b, lbfgs_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    m = 2
    H0 = np.ones_like(b, dtype=np.float32)
    x = np.zeros_like(b, dtype=np.float32)
    g = f_Ax(x) - b
    s = deque(maxlen=m)
    y = deque(maxlen=m)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for k in range(lbfgs_iters):
        # if verbose: print(fmtstr % (k, np.linalg.norm(g), np.linalg.norm(x)))
        # two-loop step
        q = g.copy()
        rho_list = []
        alpha_list = []
        for i in range(len(y)):
            rho = 1. / (y[i].dot(s[i]))
            alpha = rho * s[i].dot(q)
            q -= alpha * y[i]
            rho_list.append(rho)
            alpha_list.append(alpha)

        r = H0*q
        for i in range(len(y)):
            beta = rho_list[i] * y[i].dot(r)
            r += s[i] * (alpha_list[i] - beta)

        xnew = x - r
        gnew = f_Ax(xnew) - b
        s.append(xnew - x)
        y.append(gnew - g)

        x = xnew.copy()
        g = gnew.copy()

        if np.linalg.norm(g) < residual_tol:
            # if verbose: print(fmtstr % (k, np.linalg.norm(g), np.linalg.norm(x)))
            return x

        H0 = s[-1].dot(y[-1]) / (y[-1].dot(y[-1])) * np.ones_like(b)

    if verbose: print(fmtstr % (k, np.linalg.norm(g), np.linalg.norm(x)))
    return x



def backtrack(x, g, d, f_Ax, b, step, xp, max_linesearch=10):
    count = 0
    dec = 0.5
    inc = 2.1
    ftol=1e-4
    wolfe=0.9
    min_step=1e-20
    max_step=1e20
    fx = 0.5*x.dot(f_Ax(x)) - b.dot(x)
    result = {'status':0,'fx':fx,'step':step,'x':x,'g':g}
    # Compute the initial gradient in the search direction.
    dginit = g.dot(d)
    # Make sure that s points to a descent direction.
    if 0 < dginit:
        print('[ERROR] not descent direction')
        result['status'] = -1
        return result
    # The initial value of the objective function. 
    finit = fx
    dgtest = ftol * dginit
    while True:
        x = xp
        x = x + d * step
        # Evaluate the function and gradient values. 
        # this would change g
        Ax = f_Ax(x)
        fx = 0.5*x.dot(Ax) - b.dot(x)
        g = Ax - b
        print("[INFO]end line evaluate fx = {} step = {}.".format(fx, step))
        count = count + 1
        # chedck the sufficient decrease condition (Armijo condition).
        if fx > finit + (step * dgtest):
            print("[INFO]not satisfy sufficient decrease condition.")
            width = dec
        else:
            # check the wolfe condition
            # now g is the gradient of f(xk + step * d)
            dg = g.dot(d)
            if dg < wolfe * dginit:
                print("[INFO]dg = {} < wolfe * dginit = {}".format(dg, wolfe * dginit))
                print("[INFO]not satisfy wolf condition.")
                width = inc
            else:
                # check the strong wolfe condition
                if dg > -wolfe * dginit:
                    print("[INFO]not satisfy strong wolf condition.")
                    width = dec
                else:
                    result = {'status':0, 'fx':fx, 'step':step, 'x':x, 'g':g}
                    return result
        if step < min_step:
            result['status'] = -1
            print('[ERROR] the linesearch step is too small')
            return result
        if step > max_step:
            result['status'] = -1
            print('[ERROR] the linesearch step is too large')
            return result
        if max_linesearch <= count:
            print('[INFO] the iteration of linesearch is many')
            result = {'status':0, 'fx':fx, 'step':step, 'x':x, 'g':g}
            return result
        # update the step
        step = step * width 