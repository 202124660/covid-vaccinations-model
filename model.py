#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 02:38:23 2021

@author: bertie
"""

import sys
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from metromin import metromin
import datetime
from scipy.stats import gamma
from covid_data import *
from datetime import datetime, timedelta
from timeit import default_timer as timer
     
#######################################################################
class PopDyn(metromin) : 
    def __init__(self,Totpop,I0,Tmax,R,Kf,nP,wave, contact_matrix, priority, usepriority=True):
        """ Initialisation
            : wave : 1/2 (1st/2nd wave)
            : TotPop : total population size
            : I0 : intial number of cases
            : Tmax : number of days
            : asize : number of days +1
            : d : length of infection
            : Rpar : R-value
            : Kf : fatality rate
            : nP : padding -1
            : nPad : padding (memory size, length of dists)
            : delay : number of days to look back at for 2nd wave
            : exp_Fat : expected fatalities
            : exp_data : deaths data input
            : S : susceptible
            : I : infected
            : dose1 : first vaccine dose
            : R : recovered
            : Fat : dead
            : Di : new infections
            : Ddose1 : new 1st dose vaccinations
            : Dr : new recoveries
            : Df : new deaths
            : t : array of length Tmax+1
            : Pinf : prob dist of infecting others
            : Pinc : prob dist of showing symptoms
            : Princ : prob dist of recovering after incubation (symptoms)
            : Prd : prob dist of recovering after infection
        """
        super().__init__(10)
#        self.N0 = N0
#        self.K = 1
#        self.Ns = 1
#        self.Y = 0
        self.wave=wave
        self.TotPop = Totpop
        self.I0 = I0
        self.Tmax = Tmax
        self.asize=Tmax+1
        
        # self.d= d
        self.Rpar = R
        self.Kf = Kf
        self.nP=nP
        self.nPad = nP+1
        self.delay = 0
        self.exp_Fat =[]
        self.exp_data = []
    
        self.S=np.zeros((18, self.asize))

        self.I=np.zeros((18, self.asize))
        self.dose1=np.zeros((18, self.asize))

        self.R=np.zeros((18, self.asize))
        self.Fat=np.zeros((18, self.asize))
        
        self.Di=np.zeros((18, self.asize+self.nPad))
        self.Ddose1=np.zeros((18, self.asize+self.nPad))
        self.Dr=np.zeros((18, self.asize))
        self.Df=np.zeros((18, self.asize))
        
        self.t=np.linspace(0,self.Tmax,num=self.Tmax+1)
        
        self.Pinf=self.mk_gamma(self.t,6.5,0.62)
        self.Pinc=self.mk_gamma(self.t,5.1,0.86)
        self.Princ=self.mk_gamma(self.t,18.8,0.45)
        self.Prd = self.evalPrd()
        
        self.cmx=contact_matrix
        self.priority = priority
        self.usepriority = usepriority
         
    def intialise(self,wave, delay=14):
        if wave==1:
            for i in range(18):
                self.S[i][0]=self.TotPop[i]-self.I0
                self.I[i][0]=self.I0
                self.Di[i][self.nPad-1]=self.I0
        
        # if wave==2:
        #     self.delay = delay
        #     for d in range(1, self.delay):
        #     # exp_Fat[d] : exp fatalities on day 0-delay+d 
        #         self.Di[self.nPad-self.delay+d]=(self.exp_Fat[d+1]-self.exp_Fat[d])/self.Kf 
        #     self.Fat[0]=self.exp_data[0]
        #     self.R[0] = self.Fat[0]/self.Kf*(1-self.Kf)
        #     self.I[0] = np.sum(self.Di[self.d-self.delay:self.d])
        #     self.S[0] = self.TotPop-self.I[0]-self.R[0]-self.Fat[0] 
    
    def mk_gamma(self,t, mu, var_coef):
        """ Create and return a varience distribution
            : param t : range of values
            : param mu : average
            : param var_coef : coefficient of variation (=sigma/mu)
        """
        lambda_p = np.sqrt(1/(mu*var_coef**2)) # lambda=alpha/mu=1/(mu V^2)
        alpha_p = mu*lambda_p
        Pi = gamma.pdf(t[:self.nPad], alpha_p, 0, 1/lambda_p)
        Pi = Pi/sum(Pi) # normalise
        return(Pi)
       
       
    def evalPrd(self):
        Pr=np.zeros(self.nPad)
        for d in range(len(Pr)):
            for n in range(0,self.nP):
                Pr[d] += self.Pinc[n]*self.Princ[d-n]
        Pr = Pr/sum(Pr)     # normalise the probability    
        return Pr
    
    def larger_than(self,t,f,val=1):
        """ Take 2 arrays or lists of same size and return only the 
            one for which f is larger than val
            t : first list: only corresponbding values of f are keep
            f : values to test
            return truncated t and f as lists  
        """
        nt = []
        nf = []
        for i in range(len(f)):
            # print(len(t),len(f),i,f[i],val)
            if(f[i] > val):
                nt.append(t[i])
                nf.append(f[i])
        return(nt, nf)
 
    def read_data(self, url, country, s_date="", e_date="", col=6):
       """ Read the date file. Format
           day  N
       """
       data_list = get_data(url, country, s_date, e_date, col)
       self.exp_data = np.array(data_list)
       print(len(self.exp_data))
       
    def read_past_data(self, url, country, s_date="", e_date="", col=6):
       """ Read the date file. Format
           day  N
       """
       past_data_list = get_data(url, country, s_date, e_date, col)
       self.exp_Fat = np.array(past_data_list)
       print(len(self.exp_Fat))    
       
    def subtract_days(self, date, d):
        # returns the date d days before  "date"
        start = datetime.strptime(date, "%Y-%m-%d") #string to date
        past_date = start - timedelta(days=d) # date - days
        return(past_date.strftime("%Y-%m-%d"))
       
    def n_days(self, day1, day2):
        return int(0.49+(seconds_since_epoch(day2)-\
                         seconds_since_epoch(day1))/(3600*24))
          
    def one_step(self, vdelay=14, vprob=0.8, vfromday=0, vperday=0, dosegap=21, finalvprob=0.95, vdelay2=7, vcapacityincrease=0, vaxmax=500000):
        """ Iterate for one day.
            Logistic Equation : N(d+1) = K*N*(N-Ns)-Y*N**3*(N-Ns)**2
        """
        d = self.d
        #self.N[d+1] = self.K*self.N[d]*(self.Ns-self.N[d])\
        #               -self.Y*self.N[d]**0.5*(self.Ns-self.N[d])**2
        #if(self.N[d+1] < 0): self.N[d+1] = 0

        #self.d += 1

        if d > vfromday:
            if vperday != 0:
                # print(vperday,self.S[d],(self.TotPop-self.Fat[d]-self.vaccinated))
                num_sus_vaccinated = np.zeros(18)
                vleft = vperday + (d-vfromday)*vcapacityincrease
                if d > vfromday+dosegap:
                    vleft -= self.vaxcounthistory[d-(vfromday+dosegap)]
                if vleft > vaxmax:
                    vleft = vaxmax
                if self.usepriority == True:
                    for groupnum, i in enumerate(self.priority):
                        if groupnum == 0:
                            newlyvaxxed = 0
                        num_to_vax_in_group = max(min(vleft,self.TotPop[i]-self.Fat[i][d]-self.vaccinated[i]),0)
                        vleft -= num_to_vax_in_group
                        num_sus_vaccinated[i] = num_to_vax_in_group*self.S[i][d]/(max(self.TotPop[i]-self.Fat[i][d]-self.vaccinated[i],1))
                        self.S[i][d]=self.S[i][d]-num_sus_vaccinated[i]
                        self.dose1[i][d]=self.dose1[i][d-1]+num_sus_vaccinated[i]
                        overvax = 0
                        if self.S[i][d] < 0:
                            overvax = self.S[i][d]
                            self.S[i][d] = 0
                            if abs(overvax) > 10e-6:
                                print("overvax alert",overvax)
                        self.vaccinated[i] += num_to_vax_in_group + overvax
                        newlyvaxxed += num_to_vax_in_group + overvax
                        self.Ddose1[i][self.nPad+d]=num_sus_vaccinated[i]
                        # totSvacc = self.totSvacc[i][-1]
                        # self.totSvacc[i][d-vfromday] = num_sus_vaccinated[i] + totSvacc # check [d-vfromday]
                        # print("group",i,num_to_vax_in_group,vleft,self.S[i][d])
                else:
                    vperday = vleft
                    for i in range(18):
                        if i == 0:
                            newlyvaxxed = 0
                        perc_vax = self.TotPop[i]/sum(self.TotPop)
                        if vperday*perc_vax > vleft:
                            # print("running out",vperday*perc_vax,vleft)
                            perc_vax = vleft/vperday
                        num_to_vax_in_group = max(min(vperday*perc_vax,self.TotPop[i]-self.Fat[i][d]-self.vaccinated[i]),0)
                        vleft -= num_to_vax_in_group
                        num_sus_vaccinated[i] = num_to_vax_in_group*self.S[i][d]/(max(self.TotPop[i]-self.Fat[i][d]-self.vaccinated[i],1))
                        self.S[i][d]=self.S[i][d]-num_sus_vaccinated[i]
                        self.dose1[i][d]=self.dose1[i][d-1]+num_sus_vaccinated[i]
                        overvax = 0
                        if self.S[i][d] < 0:
                            overvax = self.S[i][d]
                            self.S[i][d] = 0
                            if abs(overvax) > 10e-6:
                                print("overvax alert",overvax)
                        self.vaccinated[i] += num_to_vax_in_group + overvax
                        newlyvaxxed += num_to_vax_in_group + overvax
                        self.Ddose1[i][self.nPad+d]=num_sus_vaccinated[i]
                        # print(i,perc_vax,num_to_vax_in_group,self.TotPop[i]/sum(self.TotPop))
                self.vaxcounthistory.append(newlyvaxxed)
                # print(self.S[:,d],self.I[:,d],self.R[:,d],self.Fat[:,d],self.vaccinated)

        SumDi = np.zeros(18)
        SumRFi = np.zeros(18)
        SumDdose1i = np.zeros(18)
        VaxxedSus = np.zeros((18, self.nPad + d + 1))
        for i in range(18):
            SumDi[i] = sum(self.Pinf*self.Di[i][self.nPad+d:self.nPad+d-self.nP-1:-1]) # total currently actually infectious (of those in the I group)
            SumRFi[i] = sum(self.Prd*self.Di[i][self.nPad+d:self.nPad+d-self.nP-1:-1]) # total of I group who just recovered/died
            # if d == vfromday + 1:
            #     print(d,i,vperday,SumDi[i],SumRFi[i])
            if vperday > 0:
                # print(vprob,finalvprob)
                # print(np.array([1-(vprob*i/(vdelay-1)) for i in range(vdelay)] + [1-vprob]*(dosegap-vdelay) + [1-vprob-((finalvprob-vprob)*i/(vdelay2-1)) for i in range(vdelay2)] + [1-finalvprob]*(self.nPad-dosegap-vdelay2)))
                # if i == 5:
                #     print(d,sum(self.Ddose1[i][self.nPad+d::-1]*np.array([1-(vprob*i/(vdelay-1)) for i in range(vdelay)] + [1-vprob]*(dosegap-vdelay) + [1-vprob-((finalvprob-vprob)*i/(vdelay2-1)) for i in range(vdelay2)] + [1-finalvprob]*(1+d+self.nPad-dosegap-vdelay2))),self.Ddose1[i][self.nPad+d::-1]*np.array([1-(vprob*i/(vdelay-1)) for i in range(vdelay)] + [1-vprob]*(dosegap-vdelay) + [1-vprob-((finalvprob-vprob)*i/(vdelay2-1)) for i in range(vdelay2)] + [1-finalvprob]*(1+d+self.nPad-dosegap-vdelay2)))
                # VaxxedSus[i] = np.flip(self.Ddose1[i][self.nPad + d::-1] * np.array([1 - (vprob * i / (vdelay - 1)) for i in range(vdelay)] + [1 - vprob] * (dosegap - vdelay) + [1 - vprob - ((finalvprob - vprob) * i / (vdelay2 - 1)) for i in range(vdelay2)] + [1 - finalvprob] * (1 + d + self.nPad - dosegap - vdelay2)))  # vaccinated who are still susceptible
                # VaxxedSus[i] = np.flip(self.Ddose1[i][self.nPad + d::-1]*self.vaxmultiplier[:self.nPad + d + 1])  # vaccinated who are still susceptible
                # VaxxedSus[i] = VaxxedSus[i][::-1]
                # SumDdose1i[i] = sum(self.Ddose1[i][self.nPad+d::-1]*np.array([1-(vprob*i/(vdelay-1)) for i in range(vdelay)] + [1-vprob]*(dosegap-vdelay) + [1-vprob-((finalvprob-vprob)*i/(vdelay2-1)) for i in range(vdelay2)] + [1-finalvprob]*(1+d+self.nPad-dosegap-vdelay2))) # total vaccinated who are still susceptible
                # SumDdose1i[i] = sum(self.Ddose1[i][self.nPad + d::-1] * self.vaxmultiplier[:self.nPad + d + 1])  # total vaccinated who are still susceptible

                VaxxedSus[i] = self.Ddose1[i][:self.nPad + d + 1]*self.vaxmultiplier[-self.nPad - d - 1:]  # vaccinated who are still susceptible
                SumDdose1i[i] = np.sum(self.Ddose1[i][:self.nPad + d + 1] * self.vaxmultiplier[-self.nPad - d - 1:])  # total vaccinated who are still susceptible
                # print(SumDdose1i,self.Ddose1[self.nPad+d:self.nPad+d-self.nP-1:-1],"\n\n")

        SumDitiled = np.tile(SumDi, (self.nPad + d + 1, 1)).T
        TotPoptiled = np.tile(self.TotPop, (self.nPad + d + 1, 1)).T
        JIVMult = self.Rpar * SumDitiled / TotPoptiled
        for i in range(18):
            if vperday > 0:
                JustInfectedVaxxed = (self.cmxtiledfull[i][:,-self.nPad-d-1:] * VaxxedSus[i] * JIVMult).sum(axis=0)
                sumJustInfectedVaxxed = np.sum(JustInfectedVaxxed)
                self.Ddose1I[i][self.nPad + d] += sumJustInfectedVaxxed  # vaxxed people who get infected
                if self.dose1[i][d] - self.Ddose1I[i][self.nPad+d] < 0:
                    # self.Ddose1I[i][self.nPad+d] = self.dose1[i][d]
                    print("ERROR 270")
                self.Di[i][self.nPad + d] += sum(self.cmx[i] * self.Rpar * self.S[i][d] * SumDi / self.TotPop) + sumJustInfectedVaxxed
                if self.S[i][d] - self.Di[i][self.nPad+d] + self.Ddose1I[i][self.nPad+d] < 0:
                    # self.Di[i][self.nPad+d] = self.S[i][d] + self.Ddose1I[i][self.nPad+d]
                    print("ERROR 274")
                self.Ddose1[i][:self.nPad + d + 1] -= JustInfectedVaxxed
            else:
                self.Di[i][self.nPad + d] += sum(self.cmx[i] * self.Rpar * self.S[i][d] * SumDi / self.TotPop)
                if self.S[i][d] - self.Di[i][self.nPad+d] < 0:
                    # self.Di[i][self.nPad+d] = self.S[i][d]
                    print("ERROR 280")

        # for i in range(18):
        #     for j in range(18):
        #         if vperday > 0:
        #             JustInfectedVaxxed = self.cmx[i][j] * self.Rpar * VaxxedSus[i] * SumDi[j] / self.TotPop[j]
        #             self.Ddose1I[i][self.nPad + d] += sum(JustInfectedVaxxed)  # vaxxed people who get infected
        #             self.Di[i][self.nPad + d] += self.cmx[i][j] * self.Rpar * self.S[i][d] * SumDi[j] / self.TotPop[j] + sum(JustInfectedVaxxed)
        #             self.Ddose1[i][:self.nPad + d + 1] -= JustInfectedVaxxed
        #         else:
        #             self.Di[i][self.nPad + d] += self.cmx[i][j] * self.Rpar * self.S[i][d] * SumDi[j] / self.TotPop[j]
        #     if i == 5:
        #         print(d,self.Ddose1I[i][self.nPad + d], self.Di[i][self.nPad + d], self.Ddose1[i][:self.nPad + d + 1],"\n\n")

        # for i in range(18):
        #     finalSumDiMult = 0
        #     for j in range(18):
        #         finalSumDiMult += self.cmx[i][j]*SumDi[j]
        #     # print(min(self.Rpar*self.S[i][d]*finalSumDiMult/self.TotPop[i],self.TotPop[i])-min(self.Rpar*(self.S[i][d]+SumDdose1i[i])*finalSumDiMult/self.TotPop[i],self.TotPop[i]))
        #     if vperday > 0:
        #         self.Ddose1I[i][self.nPad+d]=self.Rpar*SumDdose1i[i]*finalSumDiMult/self.TotPop[i]
        #         if self.dose1[i][d] - self.Ddose1I[i][self.nPad+d] < 0:
        #             self.Ddose1I[i][self.nPad+d] = self.dose1[i][d]
        #         self.Di[i][self.nPad+d]=min(self.Rpar*(self.S[i][d]+SumDdose1i[i])*finalSumDiMult/self.TotPop[i],self.TotPop[i])
        #         # self.SvaccS.append(SumDdose1i)
        #         # self.imm.append(self.totSvacc[-1]-SumDdose1i)
        #         if self.S[i][d] - self.Di[i][self.nPad+d] + self.Ddose1I[i][self.nPad+d] < 0:
        #             self.Di[i][self.nPad+d] = self.S[i][d] + self.Ddose1I[i][self.nPad+d]
        #     else:
        #         self.Di[i][self.nPad+d]=min(self.Rpar*self.S[i][d]*finalSumDiMult/self.TotPop[i],self.TotPop[i])
        #         if self.S[i][d] - self.Di[i][self.nPad+d] < 0:
        #             self.Di[i][self.nPad+d] = self.S[i][d]

            #print(i,Di[nP+i],SumDi,S[i]/TotPop)
            self.Dr[i][d]=(1-self.Kf[i])*SumRFi[i]

            self.Df[i][d]=self.Kf[i]*SumRFi[i]
        for i in range(18):
            # finalDi = 0
            # finalDdose1I = 0
            # for j in range(18):
            #     finalDi += self.cmx[i][j]*self.Di[j][self.nPad+d] # summation of self.Di[self.nPad+d] over all different groups, for i
            #     finalDdose1I += self.cmx[i][j]*self.Ddose1I[j][self.nPad+d]

            self.S[i][d+1]=self.S[i][d]-self.Di[i][self.nPad+d]
            if vperday > 0:
                self.S[i][d+1] = self.S[i][d+1] + self.Ddose1I[i][self.nPad+d]
                if self.S[i][d+1] < 0:
                    print("negative error alert",self.S[i][d+1])
                    self.S[i][d+1] = 0
                self.dose1[i][d+1]=self.dose1[i][d]-self.Ddose1I[i][self.nPad+d]
            self.I[i][d+1]=max(min(self.TotPop[i],self.I[i][d]+self.Di[i][self.nPad+d]-self.Dr[i][d]-self.Df[i][d]),0)
            self.R[i][d+1]=min(self.TotPop[i],self.R[i][d]+self.Dr[i][d])
            self.Fat[i][d+1]=min(self.TotPop[i],self.Fat[i][d]+self.Df[i][d])
        self.d += 1

    def iterate(self, d, vdelay=14, vprob=0.8, vfromday=100, vperday=0, dosegap=21, finalvprob=0.95, vdelay2=7, vcapacityincrease=0, vaxmax=500000):
        """ Iterate the equation d days
        """
        asize = d+1
        self.S=np.zeros((18, asize))
       # self.S[0]=self.TotPop-self.I0
        self.I=np.zeros((18, asize))
        self.dose1=np.zeros((18, asize))
       # self.I[0]=self.I0
        self.R=np.zeros((18, asize))
        self.Fat=np.zeros((18, asize))
        self.counterr=np.zeros((18))
        self.Di=np.zeros((18,asize+self.nPad))
        self.Ddose1=np.zeros((18,asize+self.nPad))
        self.Ddose1I=np.zeros((18,asize+self.nPad))
        self.Dr=np.zeros((18, asize))
        self.Df=np.zeros((18, asize))

        self.vaxmultiplier = np.flip(np.array([1 - (vprob * i / (vdelay - 1)) for i in range(vdelay)] + [1 - vprob] * (dosegap - vdelay) + [1 - vprob - ((finalvprob - vprob) * i / (vdelay2 - 1)) for i in range(vdelay2)] + [1 - finalvprob] * (1 + d + self.nPad - dosegap - vdelay2)))
        self.cmxtiledfull = np.zeros((18,18,self.nPad + d + 1))

        for i in range(18):
            self.cmxtiledfull[i] = np.tile(self.cmx[i], (self.nPad + d + 1, 1)).T
        # print(self.cmxtiledfull)
        self.d = 0
        
        for i in range(18):
            self.Di[i][self.nPad-1]=self.I0
        
        self.intialise(self.wave)
        
        self.vaccinated = np.zeros(18)
        # self.SvaccS = []
        # self.imm = []
        self.totSvacc = np.zeros((18, asize))
        self.vaxcounthistory = []

        for i in range(d):
            self.one_step(vdelay, vprob, vfromday, vperday, dosegap, finalvprob, vdelay2, vcapacityincrease, vaxmax)
            # if vperday > 0:
            #     if i > vfromday:
            #         self.vaccinated += vperday
            #         self.S[self.d]=self.S[self.d]-vperday
            #         self.dose1[self.d]=self.dose1[self.d-1]+vperday
            #         if self.S[self.d] < 0:
            #             self.S[self.d] = 0
            #         if self.dose1[self.d] < self.TotPop:
            #             self.dose1[self.d] = self.TotPop
        # plt.plot(self.vaxcounthistory)
        # plt.savefig('graphics/vax-capacity-changes.pdf', dpi=400, bbox_inches="tight", transparent=True)
        # plt.show()
        self.final_stats = [self.S[:,-1], self.I[:,-1], self.R[:,-1], self.Fat[:,-1], self.vaccinated,self.Fat]
        # print(self.Ddose1)
        # plt.plot(self.SvaccS,"b")
        # plt.plot(self.imm,"g")
        # plt.show()
      
    def F(self, p):
        # p : array of parameters to fit
        #self.K = p[0]
        #self.Ns = p[1]
        #self.Y = p[2]
        self.I0 = p[0]
        self.Rpar = p[1]
        Nd = len(self.exp_data)
        self.iterate(Nd)
        E = 0
        for d in range(Nd):
            if self.exp_data[d]>0 and self.Fat[d]>0:
                E += (np.log(self.exp_data[d])-np.log(self.Fat[d]))**2
        return(E)
      
    def plot(self, N=-1, e_date=""):

        Nd = len(self.exp_data)
        dates = list(reversed([datetime.strptime(e_date,'%Y-%m-%d') - timedelta(days=x) for x in range(Nd)]))
        
        plt.xlabel("Date")
        plt.ylabel("Fatalities")
        data = self.exp_data
        if(N<0):
            N = len(data)
        if N > len(self.Fat):
            N = len(self.Fat)
        if N > len(data):
            N = len(data)
            
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        if Nd < 115:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        elif Nd < 230:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
        elif Nd < 345:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
        else:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
        plt.minorticks_on()

        plt.semilogy(dates,data[:N],"r*")
        dateswithoutblanks,d = self.larger_than(dates,self.Fat.sum(axis=0)[:N],1) # remove data < 1 

        plt.semilogy(dateswithoutblanks,d,"b-")
        plt.gcf().autofmt_xdate()
        plt.show()
        #print(data[:N])
        #print(d,"\n")
        #print(self.Fat[:N])
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        if Nd < 115:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        elif Nd < 230:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
        elif Nd < 345:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
        else:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
        plt.minorticks_on()
        
        dailydata, dailymodeldata = self.cumulative_to_daily(data[:N], d)
        plt.semilogy(dates[1:],dailydata,"r*")
        plt.semilogy(dateswithoutblanks[1:],dailymodeldata,"b-")
        plt.gcf().autofmt_xdate()
        plt.show()
      
    def plot_model(self, N=-1, e_date="", Nd="", showdates=False, both=False, remove_smaller_than=-1):

        dates = list(reversed([datetime.strptime(e_date,'%Y-%m-%d') - timedelta(days=x) for x in range(Nd)]))

        plt.xlabel("Days")
        plt.ylabel("Fatalities")

        if N == -1:
            dateswithoutblanks, d = self.larger_than(dates, self.Fat.sum(axis=0), remove_smaller_than)  # remove data < 1
        else:
            dateswithoutblanks,d = self.larger_than(dates,self.Fat.sum(axis=0)[:N],remove_smaller_than) # remove data < 1

        if showdates == True:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            if Nd < 115:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
            elif Nd < 230:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
            elif Nd < 345:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
            else:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
            plt.minorticks_on()
            
            plt.plot(dateswithoutblanks,d,"b-")
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(d,"b-",linewidth=3)

        dailymodeldata, dailydata = self.cumulative_to_daily(d, "")
        
        if both == True:
            self.vaccinated_stats = [self.S[:,-1], self.I[:,-1], self.R[:,-1], self.Fat[:,-1], self.vaccinated, self.Fat]
            self.intialise(self.wave,self.delay)
            self.iterate(N)
            if N == -1:
                dateswithoutblanks,d = self.larger_than(dates,self.Fat.sum(axis=0),remove_smaller_than) # remove data < 1
            else:
                dateswithoutblanks,d = self.larger_than(dates,self.Fat.sum(axis=0)[:N],remove_smaller_than) # remove data < 1
            if showdates == True:
                plt.plot(dateswithoutblanks,d,"r--",linewidth=3)
            else:
                plt.plot(d,"r--",linewidth=3)

        # plt.savefig("prio.png",bbox_inches="tight",dpi=300)
        plt.show()
        #print(data[:N])
        #print(d,"\n")
        #print(self.Fat[:N])

        # dailymodeldata, dailydata = self.cumulative_to_daily(d, "")
        plt.xlabel("Days")
        plt.ylabel("Fatalities")
        if showdates == True:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            if Nd < 115:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
            elif Nd < 230:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
            elif Nd < 345:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=15))
            else:
                plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=20))
            plt.minorticks_on()
            
            plt.semilogy(dateswithoutblanks[1:],dailymodeldata,"b-")
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(dailymodeldata,"b-",linewidth=3)
            
        if both == True:
            dailymodeldata, dailydata = self.cumulative_to_daily(d, "")
            if showdates == True:
                plt.semilogy(dateswithoutblanks[1:],dailymodeldata,"r--",linewidth=3)
            else:
                plt.plot(dailymodeldata,"r--",linewidth=3)
        # plt.savefig("prio-cum.png",bbox_inches="tight",dpi=300)
        plt.show()
        
    def cumulative_to_daily(self, var1, var2):
        nvar1 = []
        nvar2 = []

        count = 0
        for i in var1[1:]:
            toadd = i-var1[count]
            if toadd < 0.5:
                toadd = 0.5
            nvar1.append(toadd)
            count += 1
        if var2 != "":
            count = 0
            for i in var2[1:]:
                toadd = i-var2[count]
                if toadd < 0.5:
                    toadd = 0.5
                nvar2.append(toadd)
                count += 1

        return(nvar1, nvar2)
    
    def stats(self):
        return(self.final_stats)
        # return(self.S[-1], self.I[-1], self.R[-1], self.Fat[-1], self.vaccinated)
    
    def vstats(self):
        return(self.vaccinated_stats)

if __name__ == "__main__":
    data_file = "data.txt"        
    url = "https://covid.ourworldindata.org/data/jhu/full_data.csv"
    
    TotPop = 6e7
    I0 = 1.
    Tmax = 90
    asize=Tmax+1
    d= 21
    R = 2
    Kf = 0.01
    nP=80
    country = "Brazil"
    s_date="2020-03-05"
    e_date="2020-05-05"
    wave = 1
    delay = 14
    ndays=int(0.49+(seconds_since_epoch(e_date)-seconds_since_epoch(s_date))/(3600*24))
    #print("ndays=",ndays)
    #print(list(reversed([datetime.strptime(e_date,'%Y-%m-%d') - timedelta(days=x) for x in range(ndays+1)])))
    pop = PopDyn(TotPop,I0,Tmax,d,R,Kf,nP,wave)
    
    pop.set_bounds(min=[0,0],max=[50000,5])
    pop.read_data(url, country, s_date,e_date)
    pop.read_past_data(url, country, pop.subtract_days(s_date,delay), s_date)
    # pop.read_data(data_file)
    print(pop.exp_data)
    pop.N0 = pop.exp_data[0]
    #pop.intialise(1)
    pop.intialise(wave,delay)
    
    p = pop.relax(a0=[2.,2.], T0=0.001, Tmin=1e-6, dTcoef=0.90, Nsweep=100,
                    kick0=0.1, srate=0.3, ns=10)
    print("I0={}, K={}".format(*p))
    pop.I0 = p[0]
    pop.Rpar = p[1]
    pop.iterate(len(pop.exp_data))
    pop.plot(ndays+1, e_date)

