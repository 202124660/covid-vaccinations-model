import math
import numpy as np
import random as rnd

class metromin:

    def __init__(self, trace_n=10):
        self.a0 = [0]
        self.trace_n = trace_n # tracing step no
        self.min = None
        self.max = None

    def set_bounds(self, min, max):
        self.min = min
        self.max = max

    def relax_local(self, a0, T0, Tmin, dTcoef, Nsweep, kick0, srate, ns):
        """
        param : a0 : initial value for parameters
        param : T0 : initial temperature relative to initial Energy 
        param : Tmin :  final Temperature
        param : dTcoef : temperature adjustment coeff.
        param : Nsweep : number trials for each T
        param : kick0 : initial kick size
        param : srate : rate of success -> adjust kick
        param : ns :  number of sweeps between rate test
        Uses the function dF to compute the energy variation
        """
        self.a = np.array(a0, dtype='float64')
        size_a = len(a0)
        self.T = self.F(self.a)*T0
        self.Tmin = Tmin
        self.dTcoef = dTcoef
        self.Nsweep = Nsweep
        self.kick = kick0
        self.srate = srate
        
        nsuccess = 0
        E = self.F(self.a)
        trace_n = 0
        while(self.T> self.Tmin):
            for n in range(Nsweep):
              for i in range(size_a):
                  dE0 = self.dF(i)
                  ai_old = self.a[i]
                  self.a[i] += (rnd.random()*2.-1.)*self.kick
                  # ensure we don't go beyond set bounds
                  if(self.min != None):
                      if(self.a[i] < self.min[i]):
                         self.a[i] = self.min[i] 
                  if(self.max != None):
                      if(self.a[i] > self.max[i]):
                         self.a[i] = self.max[i] 
                          
                  dE1 = self.dF(i)
                  dE = dE1-dE0
                  if(dE<0 or math.exp(-dE/self.T) > rnd.random()):
                      E += dE1-dE0
                      nsuccess += 1
                  else: # can't accept the kick => restore old value
                     self.a[i] = ai_old
                     
              if (n%ns == 0):
                # check the rate and adjust the kick
                coef = (nsuccess/(size_a*ns*self.srate))**0.25
                # avoid over scaling
                if(coef < 0.5): coef = 0.5 
                if(coef > 2): coef = 2
                # No need for small changes
                if(0.9 < coef < 1.1) : coef = 1
                self.kick *= coef
               
                nsuccess = 0
                # recompute E every now and then
                E = self.F(self.a)
            # follow what is going on
            if(trace_n == self.trace_n):
               print("T={}, E={} kick={} p=".format(self.T,E,self.kick),self.a)
               trace_n = 0
            else:
               trace_n += 1   
            # decrase the temperature
            self.T *= self.dTcoef 
        return(self.a)

    def relax(self, a0, T0, Tmin, dTcoef, Nsweep, kick0, srate, ns):
        """
        param : a0 : initial value for parameters
        param : T0 : initial temperature 
        param : Tmin :  final Temperature relative to initial Energy 
        param : dTcoef : temperature adjustment coeff.
        param : Nsweep : number trials for each T
        param : kick0 : initial kick size
        param : srate : rate of success -> adjust kick
        param : ns :  number of sweeps between rate test
        Uses the function F to compute the energy variation
        """
        self.a = np.array(a0, dtype='float64')
        size_a = len(a0)
        nsuccess = np.zeros(size_a, dtype='int')
        self.kick = np.zeros(size_a, dtype='float64')+kick0
        coef = np.zeros(size_a, dtype='float64')
        self.T = self.F(self.a)*T0
        self.Tmin = Tmin
        self.dTcoef = dTcoef
        self.Nsweep = Nsweep
        self.srate = srate
        
        self.E = self.F(self.a)
        print("E=",self.E)
        trace_n = 0
        while(self.T> self.Tmin):
            for n in range(Nsweep):
              for i in range(size_a):
                  ai_old = self.a[i]
                  self.a[i] += (rnd.random()*2.-1.)*self.kick[i]
                  # ensure we don't go beyond set bounds
                  if(self.min != None):
                      if(self.a[i] < self.min[i]):
                         self.a[i] = self.min[i] 
                  if(self.max != None):
                      if(self.a[i] > self.max[i]):
                         self.a[i] = self.max[i] 

                  Enew = self.F(self.a)
                  dE = Enew-self.E
                  if(dE<0 or math.exp(-dE/self.T) > rnd.random()):
                      self.E = Enew
                      nsuccess[i] += 1
                  else: # can't accept the kick => restore old value
                     self.a[i] = ai_old
                     
              if (n%ns == 0):
                for i in range(size_a):
                  # check the rate and adjust the kick
                  coef[i] = (nsuccess[i]/(ns*self.srate))**0.25
                  # avoid over scaling
                  if(coef[i] < 0.5): coef[i] = 0.5 
                  if(coef[i] > 2): coef[i] = 2
                  # No need for small changes
                  if(0.9 < coef[i] < 1.1) : coef[i] = 1
                  #print(self.kick[i], coef[i])
                  self.kick[i] *= coef[i]
                  if(self.max and self.kick[i] > self.max[i]):
                     self.kick[i] = self.max[i]
               
                  nsuccess[i] = 0
            # follow what is going on
            if(trace_n == self.trace_n):
               print("T={}, E={} kick={} p=".format(self.T,self.E,self.kick),self.a)
               self.show_intermediate()
               trace_n = 0
            else:
               trace_n += 1   
            # decrase the temperature
            self.T *= self.dTcoef
            self.update()
        return(self.a)

    
    def F(self, a):
        """
        : param : a : parameter set
        Compute the energy for a.
        """ 
        pass
    
    def dF(self, i):
        """
        : param : i : index of parameter 
        Compute the energy contribution of a[i]
        """
        pass

    def update(self):
        """
        Perform an update after every change of T
        """ 
        pass

    def show_intermediate(self):
        """
        Display some intermediate information
        """
        return
    
if __name__ == "__main__":

     class my_metromin(metromin):
         def __init__(self, a0, trace_n=10):
             super().__init__(trace_n)
             self.p = np.array(range(len(a0)))
             
         def F(self, a):
             """
             : param : a : parameter set
             Compute the energy for a.
             """
             da = self.a-self.p
             return(da.dot(da))

         def dF(self, i):
             """
             : param : i : index of parameter 
             Compute the energy contribution of a[i]
             """
             return((self.a[i]-self.p[i])**2)

     a0 = [0,0,0,0,0]   
     mmin = my_metromin(a0,100)
     mmin.relax(a0, 1., 1e-9, 0.99, 100, 1, 0.3, 2)
     print(mmin.a)
     
