import pandas as pd
import numpy as np
import copy
df=pd.read_csv("E:\\csv files\\car-price.csv")

data=df.values
x=[]
for i in range(len(data[0])-1):
    x.append(df.iloc[:,i])
target=df.iloc[:,-1]

class LinearRegression:
  def fit(self,x,y):
    if x.isnull().sum()==0:
      return
    else:
      print("NullCheck:Null values finded")
 
  def variance(self,x):  
     
      var1=[]    
      for i in range(len(data[0])-1):
          for j in range(len(df)):
              
              var=(((x[i][j])-x[i].mean())**2)/len(df)
              var1.append(var)

      sim_var1=copy.deepcopy(var1)
      res_var1=[]
      for i in range(len(data[0])-1):
        res_var1.append(sim_var1[0:len(df)])
        del sim_var1[0:len(df)]
      real_var1=[sum(i) for i in res_var1]
      return real_var1
 
  def covariance(self,x,y):
    
      covar1=[]
      covar=0
      for i in range(len(data[0])-1):
          for j in range(len(df)):

              covar=((x[i][j])-x[i].mean())*(y[j]-y.mean())/len(df)
              
              covar1.append(covar)
      
  
      sim=copy.deepcopy(covar1)
      res=[]
      for i in range(len(data[0])-1):
          res.append(sim[0:len(df)])
          del sim[0:len(df)]

      real_covar1=[sum(i) for i in res]
      real_covar1
      return real_covar1

  def coefficients(self,x,y):
    variance=self.variance(x)
    covariance=self.covariance(x,y)
    slope=[]
    intercept=[]
    for i in range(len(variance)):
      m=covariance[i]/variance[i]
      slope.append(m)

    for i in range(len(variance)):
      c=y.mean()-m*x[i]
      intercept.append(c.mean())

    return slope,intercept
  def predict(self,x,y,data):
    slope,intercept=self.coefficients(x,y)
  
    prediction=[]
    for i in range(len(slope)):
      y=slope[i]*data[i]+intercept[i]
      prediction.append(y)
    return sum(prediction)/len(prediction)
      
    
    
lr=LinearRegression()    
#print("Variance:",lr.variance(x))
#print("Covariance:",lr.covariance(x,target))
slope,intercept=lr.coefficients(x,target)
#print("Slope:",slope)
#print("Intercept:",intercept)
predictions=lr.predict(x,target,[3, 27, 0, 1, 0, 0, 2, 2, 0, 10, 29, 16, 13, 115, 5, 4, 28, 5, 1, 27, 13, 20, 10, 6, 9])
print(predictions)

    
