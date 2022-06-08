import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from keras.datasets import mnist
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from scipy.stats import entropy
from math import log, e
import seaborn as sns




def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log(c_normalized))  
    return H



def entropy(vec, base=2):
        " Returns the empirical entropy H(X) in the input vector."
        _, vec = np.unique(vec, return_counts=True)
        prob_vec = np.array(vec/float(sum(vec)))
        if base == 2:
                logfn = np.log2
        elif base == 10:
                logfn = np.log10
        else:
                logfn = np.log
        return prob_vec.dot(-logfn(prob_vec))



def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI




main_df = pd.read_csv('sonar.all-data.csv',header=None)
#print(main_df)
dataset = main_df.values
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]
"""
print("Lets check")
print(X)
print(Y)
"""
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
print(encoded_Y)

Y1=encoded_Y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size=0.30, random_state=42)

npX_train = X_train
npX_test = X_test
npy_train= y_train
npy_test = y_test
"""
npX_train = X_train.to_numpy()
npX_test = X_test.to_numpy()
npy_train = y_train.to_numpy
npy_test = y_test.to_numpy
print(type(npX_train))
"""


"""
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
"""

trrows,trcols = npX_train.shape
tstrows,tstcols =npX_test.shape
print(trrows)
print(trcols)
print(tstrows)
print(tstcols)
rfv=30

x_train_sample = np.empty((trrows,rfv))
x_test_sample  = np.empty((tstrows,rfv))

for i in range(0, trrows):
    temp_xtrt  = npX_train[i]
    x_train_sample[i]=  np.random.choice(temp_xtrt,rfv)

for j in range(0, tstrows):
    temp_xtst= npX_test[j]
    x_test_sample[j]= np.random.choice(temp_xtst,rfv)
"""
print(x_train_sample.shape)
print(x_test_sample.shape)
print("original array")
print(npX_train)
print("\n Reduced array")
print(x_train_sample)
"""

# Computation of dynamic range
n_neighbors =3

Fscaler = MinMaxScaler()
FX_tr_mm = Fscaler.fit_transform(npX_train)
FX_tst_mm = Fscaler.fit_transform(npX_test)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(FX_tr_mm, y_train)
acc_knn_fminmax = knn.score(FX_tst_mm, y_test)
print('Accuracy MinMax Scalar with full FV=61 KNN: {}%'.format(acc_knn_fminmax * 100))


Rscaler = MinMaxScaler()
X_tr_mm = Rscaler.fit_transform(x_train_sample)
X_tst_mm = Rscaler.fit_transform(x_test_sample)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_tr_mm, y_train)
acc_knn_rminmax = knn.score(X_tst_mm, y_test)
print('Accuracy MinMax Scalar with reduced FV=30 KNN: {}%'.format(acc_knn_rminmax * 100))


Fstd = StandardScaler()
FstdX_tr_mm = Fstd.fit_transform(npX_train)
FstdX_tst_mm = Fstd.fit_transform(npX_test)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(FstdX_tr_mm, y_train)
acc_knn_fstd = knn.score(FstdX_tst_mm, y_test)
print('Accuracy Standard Scalar with full FV =61 KNN: {}%'.format(acc_knn_fstd * 100))

Rstd = StandardScaler()
RstdX_tr = Rstd.fit_transform(x_train_sample)
RstdX_tst = Rstd.fit_transform(x_test_sample)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(RstdX_tr, y_train)
acc_knn_rstd = knn.score(RstdX_tst, y_test)
print('Accuracy Standard Scalar with reduced full FV =30 KNN: {}%'.format(acc_knn_rstd * 100))


FRbst = RobustScaler()
FrbstX_tr = FRbst.fit_transform(npX_train)
FrbstX_tst = FRbst.fit_transform(npX_test)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(FrbstX_tr, y_train)
acc_knn_frbst = knn.score(FrbstX_tst, y_test)
print('Accuracy Robust Scalar with full FV=61 KNN: {}%'.format(acc_knn_frbst * 100))

RRbst = RobustScaler()
RrbstX_tr = RRbst.fit_transform(x_train_sample)
RrbstX_tst = RRbst.fit_transform(x_test_sample)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(RrbstX_tr, y_train)
acc_knn_rbst = knn.score(RrbstX_tst, y_test)
print('Accuracy Robust Scalar with reduced full FV =30 KNN: {}%'.format(acc_knn_rbst * 100))

# (2)  Varience & (3) Entropy based feature reduction

selector=VarianceThreshold(threshold=0.02)
X_train_vrth=selector.fit_transform(X_train)
n_features1=X_train_vrth.shape[1]
print('Features after variance threshold %d with full FV',n_features1)

X_train_vr = selector.transform(X_train)
X_test_vr  =  selector.transform(X_test)
print(X_train_vr.shape)
print(X_test_vr.shape)

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_vr, y_train)
acc_knn_vr = knn.score(X_test_vr, y_test)
print('Accuracy Varience Threshold=0.02 on full fv (FV=61) KNN: {}%'.format(acc_knn_vr * 100))

threshold=0.08
Rslct=VarianceThreshold(threshold)
X_train_Rvrth=Rslct.fit_transform(x_train_sample)
n_rfv=X_train_Rvrth.shape[1]
print('Features after variance threshold with reduced FV(31)=',n_rfv)

X_train_Rvr = Rslct.transform(x_train_sample)
X_test_Rvr  =  Rslct.transform(x_test_sample)
print(X_train_Rvr.shape)
print(X_test_Rvr.shape)


knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_Rvr, y_train)
acc_knn_Rvr = knn.score(X_test_Rvr, y_test)
print('Accuracy Varience Threshold(0.08) on Reduced fv (FV )KNN: {}%'.format(acc_knn_Rvr * 100))

# Entropy on full fv & reduced fv

r,c = npX_train.shape
ent = np.empty(c)

for i in range (c):
    ent[i]=entropy(npX_train[:,i])
print("Entropy of sonar data")
print(ent)

thresh=np.arange(6.96,7.13)
ent_slct = np.where(ent > 6.96)[0]
ent_slct1 = np.where(ent >7.13)[0]

#print(ent_slct)
#print(ent_slct.shape)
X_train_ent= npX_train[:,ent_slct]
print(X_train_ent.shape[0])
print(X_train_ent.shape[1])
X_test_ent =npX_test[:,ent_slct]
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_ent, y_train)
acc_knn_ENT = knn.score(X_test_ent, y_test)
print('Accuracy entropy with full fv [threshold >6.96] KNN: {}%'.format(acc_knn_ENT * 100))


#print(ent_slct1)
#print(ent_slct1.shape)
X_train_ent= npX_train[:,ent_slct1]
print(X_train_ent.shape[0])
print(X_train_ent.shape[1])
X_test_ent =npX_test[:,ent_slct1]
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_ent, y_train)
acc_knn_ent1 = knn.score(X_test_ent, y_test)
print('Accuracy entropy with with full fv [threshold >7.13] KNN: {}%'.format(acc_knn_ent1 * 100))

# Entropy with reduced fv
r1,c1 = x_train_sample.shape
ent1 = np.empty(c1)

for k in range (c1):
    ent1[k]=entropy(npX_train[:,k])
print("Entropy of sonar data")
print(ent1)

thresh=np.arange(6.96,7.13)
ent_slct = np.where(ent1 > 6.96)[0]
ent_slct1 = np.where(ent1 >7.13)[0]
print("Dimension")
print(ent_slct)
print(ent_slct.shape)
print(ent_slct1)
print(ent_slct1.shape)

#print(ent_slct)
#print(ent_slct.shape)
X_train_ent1= x_train_sample[:,ent_slct]
print(X_train_ent1.shape[0])
print(X_train_ent1.shape[1])
X_test_ent1 =x_test_sample[:,ent_slct]
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_ent1, y_train)
acc_knn_ent1 = knn.score(X_test_ent1, y_test)
print('Accuracy entropy with reduced fv [threshold >6.96] KNN: {}%'.format(acc_knn_ent1 * 100))

X_train_ent2= x_train_sample[:,ent_slct1]
print(X_train_ent2.shape[0])
print(X_train_ent2.shape[1])
X_test_ent2 =x_test_sample[:,ent_slct1]
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_ent2, y_train)
acc_knn_ent2 = knn.score(X_test_ent2, y_test)
print('Accuracy entropy with with reduced fv [threshold >7.13] KNN: {}%'.format(acc_knn_ent2 * 100))


# Mutual Information based feature selection

bins = 6 # 
n = X_train.shape[1]
matMI = np.zeros((n, n))

for ix in np.arange(n):
    for jx in np.arange(ix+1,n):
        matMI[ix,jx] = calc_MI(X_train[:,ix], X_train[:,jx], bins)
print("Mutual information between vectors")
print(matMI)








# MI between training sample and output lable
mi = mutual_info_classif(npX_train,y_train)
mi_1=np.sort(mi)
print("MI")
print(mi_1)
mi_1 = pd.Series(mi_1)
df = pd.DataFrame(X_train)
mi_1.index =df.columns
plt.plot(mi)
plt.show()
mi_1.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.title("MI vs Feature No")
plt.ylabel('Mutual Information')
plt.xlabel('Feature No')
plt.show()


mi_score_slctindex = np.where(mi <0.0001)[0]
print(mi_score_slctindex)

X_train_mi = npX_train[:,mi_score_slctindex]
X_test_mi = npX_test[:,mi_score_slctindex]



knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_mi, y_train)
acc_knn_mixy = knn.score(X_test_mi, y_test)
print('Accuracy MI with with Full fv [threshold <0.0001] KNN: {}%'.format(acc_knn_mixy * 100))


# Reduced FV

"""

Rmi = mutual_info_classif(x_train_sample,y_train)

Rmi_score_slctindex = np.where(mi <0.0001)[0]

X_train_rmi = x_train_sample[:,Rmi_score_slctindex]
X_test_rmi = x_test_sample[:,Rmi_score_slctindex]

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(x_train_rmi, y_train)
acc_knn_mixy = knn.score(x_test_rmi, y_test)
print('Accuracy entropy with with Reduced fv [threshold <0.0001] KNN: {}%'.format(acc_knn_mixy * 100))










X_train_MI= X_train[:,mi_score_selected_index]
X_test_MI=X_test[:,mi_score_selected_index]
#print(X_train_MI.shape[0])
#print(X_train_MI.shape[1])

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train_MI, y_train)
acc_knn_MI = knn.score(X_test_MI, y_test)

print('Accuracy MI with KNN: {}%'.format(acc_knn_MI * 100))
"""
num_comp=60
pca = PCA(n_components=num_comp)
x_reduced = pca.fit_transform(X_train)
pca_var = sum(pca.explained_variance_ratio_)
print(pca_var)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,num_comp,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

sklearn_pca = PCA(n_components=60)
xtr_Red = sklearn_pca.fit_transform(X_train)
xtst_Red = sklearn_pca.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(xtr_Red, y_train)
acc_knn_rpca = knn.score(xtst_Red, y_test)

print('Accuracy PCA with 30 components with KNN: {}%'.format(acc_knn_rpca * 100))


