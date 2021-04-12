#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:37:34 2020

@author: meg
"""
import datetime
import re
import matplotlib.pyplot as plt 
import numpy as np
import urllib.request

# data file available from
# https://covid.ourworldindata.org/data/ecdc/full_data.csv

def seconds_since_epoch(date, form='%Y-%m-%d'):
    """ Convert date into sec since 1 Jan 1970
        date : date as a string
        form : format for the string
    """
    dt = datetime.datetime.strptime(date,form)
    return(dt.timestamp())

def get_data(src, country, s_date="", e_date="", col=6):
    """ Read data from file.
        Select specific column for country between set dates
        src : url or filename
        country to select
        s_date : start date in yyyy-mm-dd format
        e_date : end date (inclusive) in yyyy-mm-dd format
        col : column to extract (first is 1)
              3: new_cases, 4: new_deaths 5: total_cases 6: total_deaths
              7: weekly_cases 8: weekly_deaths 9: biweekly_cases 
              10: biweekly_deaths

        Return data as a list
    """
    type_URL = True
    
    print("Fitting deaths in {} from {} to {}.".format(country,s_date,e_date))
    
    if (s_date==""): start_time = 0
    else: start_time = seconds_since_epoch(s_date)

    if (e_date==""): end_time = seconds_since_epoch("2030-01-01")
    else: end_time = seconds_since_epoch(e_date)

    if(src[:4]=="http"):
       fp = urllib.request.urlopen(src)
    else:  
       fp = open(src, "r")
       type_URL = False
    data = []
    dates = []
    
    firstdate = 0
    for line in fp.readlines():
          if(type_URL):
            w = line.decode('utf8').split(",")
          else:  
            w = line.split(",")
          if (w[1] == country):
              t = seconds_since_epoch(w[0])
              if (t >= start_time) and (t <= end_time):
                 if firstdate == 0:
                     firstdate = t
                 val = w[col-1]
                 if val == "":
                     val = 0
                 else:
                     val = int(float(val))
                 data.append(val)
                 dates.append(t)
    
    # fill missing early data with ecdc data if necessary
    fp1 = urllib.request.urlopen("https://covid.ourworldindata.org/data/ecdc/full_data.csv")
    count = 0
    for line in fp1.readlines():
          w = line.decode('utf8').split(",")
          if (w[1] == country):
              t = seconds_since_epoch(w[0])
              if (t >= start_time) and (t <= end_time) and (t < firstdate):
                 val = w[col-1]
                 if val == "":
                     val = 0
                 else:
                     val = int(float(val))
                 data.insert(count,val)
                 dates.insert(count,t)
                 count += 1
    if dates[0] != start_time:
        print("COULDN'T RETRIEVE INITIAL DATA. First date collected is",datetime.datetime.fromtimestamp(dates[0]).strftime("%Y-%m-%d"))
    if dates[-1] != end_time:
        print("Data collected up to",datetime.datetime.fromtimestamp(dates[-1]).strftime("%Y-%m-%d"))
    return(data)             

# example of use
if __name__ == "__main__":
  #fname = "https://covid.ourworldindata.org/data/ecdc/full_data.csv"
  fname = "covid_data_28Nov2020.csv"
  l = get_data(fname, "United Kingdom", "2020-07-01","")
  death = np.array(l)
  #death -= death[0] # up to now
  plt.xlabel("d")
  plt.ylabel("Deaths")
  plt.semilogy(death)
  plt.show()
