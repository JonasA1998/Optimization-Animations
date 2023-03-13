import numpy as np
import numdifftools as nd
import numpy.linalg as LA
def armijo(f,x,d,l=.5,alpha=1/4,max_it=30,TOL=2):
        ''' Armijo stepsize regularization. 
        
        Inputs: 
            f: function
            x: starting point for stepsize
            d: direction
        Outputs:
            tau: stepsize >0
        '''
        tau = 1
        j=0
        if(nd.Gradient(f)(x).shape==()):
            dtd=d@nd.Gradient(f)(x)
        else:
            dtd=d.T@nd.Gradient(f)(x)
        while(abs(f(x+tau*d)>f(x)+alpha*tau*dtd)):
            #print(f(x+tau*d),f(x)+alpha*tau*d.T@nd.Gradient(f)(x))
            if(j==max_it):
                #print("maxit")
                break
            taustar=-.5*(tau**2)*dtd/(f(x+tau*d)-f(x)-tau*dtd)
            tau=max(l*tau,taustar)
            j+=1
        return tau
class opti:
    #Gradient Descent with or without stepsize
    

    def grad_des(h,x0,TOL,max_it,stepreg=False):
        '''' Gradient Descent for unconstraint Optimization
        
        Inputs: 
            h: function
            x0: starting point of shape(n,1)
            TOL: tolerance for convergence
            max_it: maximum iteration for convergence
            stepreg: if true armijo, else sk = 1
        Outputs:
            x0: starting point
            x_out: vector of shape (n,m)
            maxlen: max length over all n for animation
        '''
        maxlen=0
        x_out=[]
        for i in range(len(x0)):
            sk=1
            ite=0
            x_act=x0[i]
            x_ar=[x_act]
            while(ite<max_it):
                if(np.linalg.norm(nd.Gradient(h)(x_act))<TOL):
                    print("GD converged at iteration:%d"%ite)
                    break
                ite+=1
                if(stepreg==False):
                    sk=1
                else:
                    sk=armijo(h,x_act,-nd.Gradient(h)(x_act))
                x_act=x_act-sk*nd.Gradient(h)(x_act)
                x_ar.append(x_act)
            if(len(x_ar)>maxlen):
                maxlen=len(x_ar)
            if(ite==max_it):
                print("GD reached iteration limit:%d"%ite)
            x_out.append(x_ar)
        return x0,x_out,maxlen
    #Newton with or without stepsize
    def ungedNewt(f,x0,TOL,max_it,stepreg=False):
        '''ungedaempft/gedaempft Newton for unconstraint Optimization
        
        Inputs: 
            f: function
            x0: starting point of shape(n,1)
            TOL: tolerance for convergence
            max_it: maximum iteration for convergence
            stepreg: if true armijo, else sk = 1
        Outputs:
            x0: starting point
            x_out: vector of shape (n,m)
            maxlen: max length over all n for animation
        '''
        x_out=[]
        maxlen=0
        sk=1
        for i in range(len(x0)):
            ite=0
            x_act=x0[i]
            x_ar=[x_act]
            while(ite<max_it):
                ite+=1
                if(stepreg==False):
                    sk=1
                else:
                    sk=armijo(f,x_act,-LA.inv(nd.Hessian(f)(x_act))@nd.Gradient(f)(x_act))
                x_act=x_act-sk*LA.inv(nd.Hessian(f)(x_act))@nd.Gradient(f)(x_act)
                x_ar.append(x_act)
                if(np.linalg.norm(nd.Gradient(f)(x_act))<TOL):
                    print("Newt converged at iteration:%d"%ite)
                    break
            if(len(x_ar)>maxlen):
                    maxlen=len(x_ar)
            if(ite==max_it):
                print("Newt reached iteration limit:%d"%ite)
            x_out.append(x_ar)
        return x0,x_out,maxlen