import pandas as pd
df=pd.read_csv(r"D:\SANKHYANA\Practise Data\logistic_sample_data.csv")

from sklearn.metrics import *
train=df.iloc[:int(0.7*len(df))]
test=df.iloc[int(0.7*len(df)):]

X_train=train.iloc[:,:11]
Y_train=train.iloc[:,11]

X_test=test.iloc[:,:11]
Y_test=test.iloc[:,11]

data=X_train.values
target=Y_train.values
t_data=X_test.values
class Logistic_Regression:
    def __init__(self,x,y):
        learning_rate=0.01
        count=0
        while count<len(x[0]):
            self.w=[0 for i in range(len(x[0]))]
            self.predictions=[1 for i in range(len(x))]
            for i in range(len(x)):
                
                f=np.dot(np.transpose(self.w),x[i])
                f=(1/(1+np.exp(-f)))   #sigmoid function
                z=0
                if f>z:
                    yhat=1
                else:
                    yhat=0
                    
                for j in range(len(self.w)):
                    self.w[j]=self.w[j]+(y[i]-yhat)*learning_rate*x[i][j]
            count+=1        
            sse=[]
            for i in range(len(y)):
                self.predictions[i]=(y[i]-yhat) 
            sse.append((0.5)*sum(self.predictions))
       
        print(classification_report(y,self.predictions))
                
       
    def pred_vals(self,x):
        count=0 
        print(self.w)
        while count<len(x[0]):            
            self.pred=[0 for i in range(len(x))]
            for i in range(len(x)):
                
                f=np.dot(np.transpose(self.w),x[i])
                f=(1/(1+np.exp(-f)))   #sigmoid function
                z=1
                if f>z:
                    yhat=1
                else:
                    yhat=0
                self.pred.append(yhat)
            count+=1            
lr=Logistic_Regression(data,target)
lr.pred_vals(t_data)

accuracy_score(Y_test,lr.pred[:16443])
print(classification_report(Y_test,lr.pred[:16443]))
