#Aciklama:
#Beton Cokme Testi
#Betonun betonlasmadan onceki kivamini olcen test
#Betonun islenebilirligi ve akiskanligi onemli bundan dolayi beton cokme testi 
#yapiliyor(Yanlis karistirilmis betonlari ayirt etmek icin)
#9 tane feature var
#7 tanesi 1m3 betonda bulunan kg cinsinden bilesenler(Cement Slag,Fly ash,Water,SP,Coarse Aggr.Fine Aggr.)
#Cement(cimento)
#Slag(Cimentonun kaynasmasini saglayan bi madde)
#Fly ash(ucan kul):cimentonun kivami ile alakali

#(Measurements)(2 feature )
#SLUMP (cm)-cokme
#FLOW (cm)-akiskanlik 

#Target:Beton donduktan 28 gun sonra basinc testine maruz birakiliyor.
#Test sonucu bizim target imiz(Betonun ne kadar basinca dayanikli oldugu)
#28-day Compressive Strength (Mpa)


##Importing dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 100)


####Loading data and EDA

df = pd.read_csv("cement_slump.csv")
df.head()
# Cement	Slag	Fly ash	Water	SP	Coarse Aggr.	Fine Aggr.:1 kg betonda olan bileşenler 
# SLUMP(cm)	FLOW(cm) : Betonun kıvamını ölçen metrikler

df.info()
#null deger yok 
#Outlier durumunu kontrol ettik 
#mean ve std lerde std nin mean den buyuk oldugu durum yok,outlier durumu yok 
#diyebiliriz

df.corr()['Compressive Strength (28-day)(Mpa)']
# Cement, Fly Ash, Slag.. Bunların korelasyonları yüksek target ile 
#Hedef sutunun diger sutunlarla olan correlasyonuna bakiyoruz 
#Pozitif yonde en yuksek corr(Cement,Fly ash)
#Negatif yondede en yuksek yuksek corr(Slag)


###Graphical analysis
plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot = True, vmin=-1, vmax=1);
# Multicollinarity var SLUMP ve FLOW arasında : 0.91 .. Bu featureların birbirini baskılama durumu var
# Lineer regression bunu halledemiyordu. Ridge ve Lasso bunları hallediyordu

plt.figure(figsize =(20,10))
df.boxplot()
# Outlier değerleri describe da değerlendirmiştik. Burada herhangi bir outlier görmüyoruz
# Genel inside elde ettik. Şimdi modellemeye geçelim

#Train | Test Split

X = df.drop("Compressive Strength (28-day)(Mpa)", axis =1)
y = df["Compressive Strength (28-day)(Mpa)"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
#%20 test datasina ayrildi(test_size=0.2)

###Scaling the Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
#Otulier durumlarda genelde RobustScaler tercih ediliyor
#Eger datada dummy edilmis featurelar varsa MinMaxScaler normal dagilimini bozmama
#adina tavsiye ediliyor(0-1 arasinda degerleri sikistirir)
#StandartScaler(-1 ile 1 arasinda degerleri sikistirir)
scaler = StandardScaler() # will be used in pipeline later
# if you don't use pipeline, you can use scaler directly
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train) 
X_test_scaled = scaler.transform(X_test)

###Pipeline


#%%%
###Linear Regression


from sklearn.pipeline import Pipeline # pipeline is used to combine scaler and model
from sklearn.linear_model import LinearRegression

lm = LinearRegression() # will be used in pipeline later
pipe_lm = Pipeline([("scaler", scaler), ("lm", lm)]) # pipeline is used to combine scaler and model
#linear regressionun pipe modeli,burda modellemeyi yapmis olduk,Tek satirda hem scalin yapildi hemde model kuruldu 
pipe_lm.fit(X_train, y_train)#Train seti uzerunden egitim yapildi
#Pipeline(steps=[('scaler', StandardScaler()), ('lm', LinearRegression())])
#Cikti olarak yapilan adimlari gosteriyor 
y_pred = pipe_lm.predict(X_test) # predict on test data
#Egitimi yaptiktan sonra hic gormedigi X_test datasinda bir tahmin aliyoruz,bunu y_pred e atadik
y_train_pred = pipe_lm.predict(X_train) 
# predict on train data
#Gordugu daha once egitim aldigi data uzerindan tahmin aliyoruz 
#karsilastirip modelin genelleme yapip yapmadigina bakacaz
#Yani overfitting var mi yok mu 


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_val(y_train, y_train_pred, y_test, y_pred, model_name):
    
    scores = {model_name+"_train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    model_name+"_test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)
ls =train_val(y_train, y_train_pred, y_test, y_pred, "linear") # train and test scores
ls
#   linear_train	linear_test
#R2	    0.907145	0.911967
#mae	1.936263	1.740383
#mse	5.874313	4.454237
#rmse	2.423698	2.110506
# Test scorum tek seferlik 0.91 yani yüksek çıkmış olabilir. Bunu CV ile test edeceğiz

###Cross Validate
#from sklearn.metrics import SCORERS
#list(SCORERS.keys())
from sklearn.model_selection import cross_validate, cross_val_score
model = Pipeline([("scaler", scaler), ("lm", lm)]) # Modelimizi 0 ladık tekrar
scores = cross_validate(model, X_train, y_train, scoring = ['r2', 'neg_mean_absolute_error','neg_mean_squared_error', \
                                                            'neg_root_mean_squared_error'], cv = 5)
#scroring icindeki negatif scorlarin sebebi,CV yaparken biz R2 scorunu maximize etmeye 
#calisiyoruz,diger yandan da hatayi minimize etmeye calismamiz lazim,ikisini ayni anda
#yapabilmek icin erroru negatif taraftan maximize ederek normalde minimize etmis oluyoruz 
# CV= maksimizasyon algoritması.. 

pd.DataFrame(scores, index = range(1,6))
#   fit_time  score_time   test_r2  test_neg_mean_absolute_error  \
#1  0.009664    0.005737  0.942208                     -1.908524   
#2  0.007790    0.005629  0.706629                     -2.647319   
#3  0.007355    0.004397  0.865934                     -1.482831   
#4  0.008304    0.006062  0.923166                     -2.244108   
#5  0.007267    0.005064  0.792889                     -2.867905   

#   test_neg_mean_squared_error  test_neg_root_mean_squared_error  
#1                    -4.645188                         -2.155270  
#2                   -10.756658                         -3.279734  
#3                    -3.725762                         -1.930223  
#4                    -8.434009                         -2.904137  
#5                   -11.698265                         -3.420273  

#test_r2 0.94 daha yuksek ,0.70 daha dusuk geldigi yerlerde var
#asagida bu scorlarin ortalamasini alacaz

scores = pd.DataFrame(scores, index=range(1,6))
scores.iloc[:, 2:].mean()
# Asıl scorumuz 0.84(test) , bunu train hatası ile karşılaştırıyoruz(0.90)
#test_r2                             0.846165
#test_neg_mean_absolute_error       -2.230137
#test_neg_mean_squared_error        -7.851977
#test_neg_root_mean_squared_error   -2.737927

train_val(y_train, y_train_pred, y_test, y_pred, "linear")
#      linear_train  linear_test
#R2        0.907145     0.911967
#mae       1.936263     1.740383
#mse       5.874313     4.454237
#rmse      2.423698     2.110506

#Tekrar bakmak icin CV oncesi scorlarimizi cagirdik.R2 91 di CV sonunda 
#gercek scorumuz 0.84 cikti, bu test scorunu train scoru ile karsilastirarak(0.90)
#overfitting var mi yok mu diye karar veriyoruz.
#Cv test scoru 0.84,train scoru 0.90,aralarinda fark var ama direk overfitting var diyemeyiz 
#Yuzdesel olarak hataya bakmamiz lazim 

print("train RMSE:", 2.423698/df["Compressive Strength (28-day)(Mpa)"].mean())
print("CV RMSE:", 2.737927/df["Compressive Strength (28-day)(Mpa)"].mean())
#train RMSE: 0.06725257718905493
#CV RMSE: 0.07597177821060941
#RMSE lere bakarak  train ve CV icin yustesel hatalari bulduk(0.06 ve 0.07)
#Burada yuzdesel scorlar birbirine yakin oldugu icin overfitting yoktur dedik
#
pipe_lm["lm"].coef_ #katsayilara baktik 
#array([ 4.28015177, -2.22287954,  3.9320198 , -4.77083923,  0.12109966,
#       -5.27729122, -2.61131342, -2.18837804,  1.46570621])
lm_df = pd.DataFrame(pipe_lm["lm"].coef_, columns = ["lm_coef"])
lm_df #coeff leri alarak dataframe olsuturduk sonradan yorumlamak icin 
lm_df
#0	4.280152
#1	-2.222880
#2	3.932020
#3	-4.770839
#4	0.121100
#5	-5.277291
#6	-2.611313
#7	-2.188378
#8	1.465706


#%%%

#####Ridge Regression



from sklearn.linear_model import Ridge
# Modele hata ekleyip bias ve varyans arasındaki dengeyi sağlıyordu
ridge_model = Ridge(alpha=1, random_state=42) # will be used in pipeline later
#Ridge modeli defaul degerleriyle calistirdik(alpha=1)
pipe_ridge = Pipeline([("scaler", scaler), ("ridge", ridge_model)]) # pipeline is used to combine scaler and model
# Pipeline sizin için scaling yapıyor. Modelin kuruyor
# Test setine transform yapıyor ve transform yapılmışı model içerisine koymuş oluyor
pipe_ridge.fit(X_train, y_train) #egitimi tamamdi

y_pred = pipe_ridge.predict(X_test) #tahmin aliyoruz,hic gormedigi data(X_test)
y_train_pred = pipe_ridge.predict(X_train) #daha once egitimini aldigi data (X_train)

rs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge")#scorlari birbirleriyle carsilastirkma icin 
#rs degiskenine atadik 
rs
#      ridge_train  ridge_test
#R2       0.906392    0.911204
#mae      1.938096    1.732472
#mse      5.921967    4.492822
#rmse     2.433509    2.119628
#Scorlar birbirlerine yakin fakat yine de CV ile bakmaliyiz,datanin guzel yeri denk
#gelmis olabilir 
pd.concat([ls, rs], axis=1)  # combine train and test scores to compare
#      linear_train  linear_test  ridge_train  ridge_test
#R2        0.907145     0.911967     0.906392    0.911204
#mae       1.936263     1.740383     1.938096    1.732472
#mse       5.874313     4.454237     5.921967    4.492822
#rmse      2.423698     2.110506     2.433509    2.119628
 

###For Ridge Regression CV with alpha : 1

model = Pipeline([("scaler", scaler), ("ridge", ridge_model)]) # Pipeline ı kurduk tekrar
scores = cross_validate(model, X_train, y_train,
                    scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)
pd.DataFrame(scores, index = range(1, 6))
#   fit_time  score_time   test_r2  test_neg_mean_absolute_error  \
#1  0.011642    0.005769  0.945011                     -1.837154   
#2  0.008090    0.005954  0.708681                     -2.651491   
#3  0.007831    0.005544  0.870580                     -1.487804   
#4  0.008562    0.005699  0.920645                     -2.255780   
#5  0.007732    0.006100  0.804676                     -2.793295   

#   test_neg_mean_squared_error  test_neg_root_mean_squared_error  
#1                    -4.419908                         -2.102358  
#2                   -10.681426                         -3.268245  
#3                    -3.596664                         -1.896487  
#4                    -8.710815                         -2.951409  
#5                   -11.032516                         -3.321523  
scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()
#r2 skorlarina bakti 0.94,0.70 gibi degerler var.Cv oncesi 0.91 di yani datanin guzel yerine denk gelmis
#skorlarin ortalamasina bakarak yorumlayacagiz

#test_r2                             0.849918
#test_neg_mean_absolute_error       -2.205105
#test_neg_mean_squared_error        -7.688266
#test_neg_root_mean_squared_error   -2.708004

#R2 skor Cv sonunda 0.84 cikti,tek severlik aldigimiz skor 0.91 di,asagida tek 
#seferlik aldigimiz skorlari cagirarak kiyaslama yaptik
#train seti skoru(0.90) ile cv test skoru(0.84) e baktik aralarinda fark gozukuyor
#ama yuzdesel olarak bakarsak burada da overfitting durumunun olmadigini goruruz  
train_val(y_train, y_train_pred, y_test, y_pred, "ridge")
#      ridge_train  ridge_test
#R2       0.906392    0.911204
#mae      1.938096    1.732472
#mse      5.921967    4.492822
#rmse     2.433509    2.119628


pipe_ridge["ridge"].coef_ #coef(katsayilara) baktik
#array([ 5.01092233, -1.37765187,  4.8062743 , -3.90644878,  0.26138511,
#       -4.06644253, -1.74660531, -1.95498663,  1.13349998])
rm_df = pd.DataFrame(pipe_ridge["ridge"].coef_, columns = ["ridge_coef_1"])
pd.concat([lm_df,rm_df], axis = 1) 
# Diğer modellerin skorlarıyla karşılaştırma yapayım
# lm_coef: Lineer regression coefficientlar
# ridge_coef_1: Alphası 1 olan coefficientlar
# Ridge Model butun featurelara bir katsayi atiyordu(Lasso onemsiz gordugu featurleri direk sifirliyordu)
# SLUMP ve FLOW arasinda multicolinearty bir durum vardi(corr:0.91)
# 7-flow,8-Slump,Linear regression multicolinerity sorununu cozmuyordu
# Lineer regression da flow daha baskılamış gibi(-2.188378 daha yüksek 1.46 ya göre yani baskılamış)
# ridge_coef_1 de değerlerin oransal değeri birbirine yaklaşmış. Ridge çözmüş sorunu
# Ridge de -1.954987 , 1.133500 birbirlerine nispeten yakınlaşmışlar yani ridge multicollinerity yi çözmüş

#    lm_coef  ridge_coef_1
#0  4.280152      5.010922
#1 -2.222880     -1.377652
#2  3.932020      4.806274
#3 -4.770839     -3.906449
#4  0.121100      0.261385
#5 -5.277291     -4.066443
#6 -2.611313     -1.746605
#7 -2.188378     -1.954987
#8  1.465706      1.133500
 
####Choosing best alpha value with Cross-Validation

from sklearn.linear_model import RidgeCV # Burada gridsearchCV de kullanabilirdik
alpha_space = np.linspace(0.1, 1, 100) # Hyperparametreyi tanımlıyorum. 100 adet değer ürettik
alpha_space
# Not: Parametre sayısı arttıkça bu kadar fazla örneklem alırsanız bilgisayarı çok kasacaktır

ridge_cv_model = RidgeCV(alphas=alpha_space, cv = 10, scoring= "neg_root_mean_squared_error") # will be used in pipeline later
# alphas=alpha_space parametreleri verdik. Grid search yapıyoruz en ideal parametreyi bulmak için
pipe_ridgecv = Pipeline([("scaler", scaler), ("ridgecv", ridge_cv_model)]) # pipeline is used to combine scaler and model
pipe_ridgecv.fit(X_train, y_train) # Eğitim yapıyorum

pipe_ridgecv["ridgecv"].alpha_ # En iyi alpha buymuş 0.91
#0.9181818181818181


##### Ridge( alpha = 0.91) model uzerinden tahmin aliyoruz 
y_pred = pipe_ridgecv.predict(X_test)
y_train_pred = pipe_ridgecv.predict(X_train)  
rcs = train_val(y_train, y_train_pred, y_test, y_pred, "ridge_cv") 
rcs
# Lineer regressionda ve ridge aldığım scora yakın görünüyor
# Bunun içinde bir cross validation yapmayacağız. Üstte yaptık zaten
#      ridge_cv_train  ridge_cv_test
#R2          0.906476       0.911355
#mae         1.937773       1.732412
#mse         5.916637       4.485191
#rmse        2.432414       2.117827


pd.concat([ls, rs, rcs], axis = 1) 
#      linear_train  linear_test  ridge_train  ridge_test  ridge_cv_train  \
#R2        0.907145     0.911967     0.906392    0.911204        0.906476   
#mae       1.936263     1.740383     1.938096    1.732472        1.937773   
#mse       5.874313     4.454237     5.921967    4.492822        5.916637   
#rmse      2.423698     2.110506     2.433509    2.119628        2.432414   

#      ridge_cv_test  
#R2         0.911355  
#mae        1.732412  
#mse        4.485191  
#rmse       2.117827  

# Diğer modellerin skorlarıyla karşılaştırma yapayım
# linear_train rmse 2.423698 den ridge_train rmse 2.433509 e çıkmış hata, yani modele hata eklemiş ridge
# Data iyi oldugu icin degisiklikler cok az bozuk datalarda bu daha cok belirir
# ridge_train	ridge_test : alpha = 0.1 iken
# Ridge_cv_train	ridge_cv_test = 0.9 --> hatayı milimetrik düşürmüş(2.432414)
# Ozetle Ridge model modele hata ekleyip skorlari birbirlerine yakinlastirmis oluyor  
#Yani multicolinearity ve overfitting sorununu cozmus oluyor 


pipe_ridgecv["ridgecv"].coef_ #Ringe icin coeff lere baktik
#array([ 5.00521265, -1.38945391,  4.79652108, -3.93170693,  0.25805802,
#       -4.09232266, -1.76329842, -1.96996865,  1.15500214])

rcm_df = pd.DataFrame(pipe_ridgecv["ridgecv"].coef_, columns=["ridge_cv_coef_0.91"])
pd.concat([lm_df,rm_df, rcm_df], axis = 1) #buldugumuz tum coeffleri concat ederek kiyaslayacagiz 

#    lm_coef  ridge_coef_1  ridge_cv_coef_0.91
#0  4.280152      5.010922            5.005213
#1 -2.222880     -1.377652           -1.389454
#2  3.932020      4.806274            4.796521
#3 -4.770839     -3.906449           -3.931707
#4  0.121100      0.261385            0.258058
#5 -5.277291     -4.066443           -4.092323
#6 -2.611313     -1.746605           -1.763298
#7 -2.188378     -1.954987           -1.969969
#8  1.465706      1.133500            1.155002



#%%

#######LASSO


from sklearn.linear_model import Lasso, LassoCV
lasso_model = Lasso(alpha=1, random_state=42)
# Bütün parametreler default ise vanilla parameter deniyor buna
# Büyük alpha büyük hata ya da büyük regularization demek(1 büyük diyebiliriz burada)
pipe_lasso = Pipeline([("scaler", scaler), ("lasso", lasso_model)]) # pipeline is used to combine scaler and model
pipe_lasso.fit(X_train, y_train)
Pipeline(steps=[('scaler', StandardScaler()),
                ('lasso', Lasso(alpha=1, random_state=42))])
y_pred = pipe_lasso.predict(X_test)
y_train_pred = pipe_lasso.predict(X_train)
lss = train_val(y_train, y_train_pred, y_test, y_pred, "lasso") 

lss
#      lasso_train  lasso_test
#R2       0.776425    0.801642
#mae      2.959616    2.466012
#mse     14.144056   10.036383
#rmse     3.760858    3.168025

# Genelde train yüksek olur teste göre
# Burada lasso bazı featureları 0 ladığı için böyle oldu
pd.concat([ls, rs, rcs, lss], axis = 1)
#      linear_train  linear_test  ridge_train  ridge_test  ridge_cv_train  \
#R2        0.907145     0.911967     0.906392    0.911204        0.906476   
#mae       1.936263     1.740383     1.938096    1.732472        1.937773   
#mse       5.874313     4.454237     5.921967    4.492822        5.916637   
#rmse      2.423698     2.110506     2.433509    2.119628        2.432414   

#      ridge_cv_test  lasso_train  lasso_test  
#R2         0.911355     0.776425    0.801642  
#mae        1.732412     2.959616    2.466012  
#mse        4.485191    14.144056   10.036383  
#rmse       2.117827     3.760858    3.168025  

# linear_train rmse 2.423698 den ridge_train rmse 2.433509 e çıkmış hata yani modele hata eklemiş ridge
# ridge_train	ridge_test : alpha = 0.1 iken
# Ridge_cv_train	ridge_cv_test = 0.9 --> hatayı milimetrik düşürmüş(2.432414)
# Hata ekleyip ...
# Bir cross validation yapalım




####For Lasso CV with Default Alpha : 1

model = Pipeline([("scaler", scaler), ("lasso", lasso_model)]) # CV öncesi modeli sıfırlamak lazım
scores = cross_validate(model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)
pd.DataFrame(scores, index = range(1, 6))
#   fit_time  score_time   test_r2  test_neg_mean_absolute_error  \
#1  0.011651    0.005705  0.826285                     -3.154233   
#2  0.007071    0.005183  0.499657                     -3.387816   
#3  0.006977    0.005464  0.676319                     -2.382057   
#4  0.007690    0.005485  0.681643                     -4.372939   
#5  0.011131    0.006980  0.736091                     -3.244457   

#   test_neg_mean_squared_error  test_neg_root_mean_squared_error  
#1                   -13.962787                         -3.736681  
#2                   -18.345441                         -4.283158  
#3                    -8.995287                         -2.999214  
#4                   -34.945941                         -5.911509  
#5                   -14.906371                         -3.860877

scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:].mean()
#test_r2                              0.683999
#test_neg_mean_absolute_error        -3.308301
#test_neg_mean_squared_error        -18.231165
#test_neg_root_mean_squared_error    -4.158288

#R2 0.68 cikti tek seferlik aldigimiz scor 0.80 di,skor dusmus oldu
#trainde 0.77 ,test de 0.68 lik skorlar var, bunlara bakarak underfitting var 
#diyebiliriz 
#Lassonun alpha parametresinin 1 oldugu durumda underfitting durumu var model ogrenemiyor 
#alpha degeri buyuk oldugu icin modele fazla hata ekliyor,fazla regilarization yapmis oluyor
#Bunu ideal alpha degerini belirleyip cozmemiz lazim 

train_val(y_train, y_train_pred, y_test, y_pred, "lasso")
#      lasso_train  lasso_test
#R2       0.776425    0.801642
#mae      2.959616    2.466012
#mse     14.144056   10.036383
#rmse     3.760858    3.168025


model["lasso"].coef_
#array([ 4.82131411, -0.,  4.88005283, -0.81976891,  0.  ,  -0.35149513, -0. , -0.71063068, -0.])
# 4 feature ı lasso model sıfırlamış
# Bakalım az feature la skorları yukarı çekebilecekmiyiz 

lsm_df = pd.DataFrame(model["lasso"].coef_, columns = ["lasso_coef_1"])
pd.concat([lm_df, rm_df, rcm_df, lsm_df], axis = 1) 
#    lm_coef  ridge_coef_1  ridge_cv_coef_0.91  lasso_coef_1
#0  4.280152      5.010922            5.005213      4.821314
#1 -2.222880     -1.377652           -1.389454     -0.000000
#2  3.932020      4.806274            4.796521      4.880053
#3 -4.770839     -3.906449           -3.931707     -0.819769
#4  0.121100      0.261385            0.258058      0.000000
#5 -5.277291     -4.066443           -4.092323     -0.351495
#6 -2.611313     -1.746605           -1.763298     -0.000000
#7 -2.188378     -1.954987           -1.969969     -0.710631
#8  1.465706      1.133500            1.155002     -0.000000
# aldığımız tüm coefleri datafırem oluşturup tekrar inceledik
#Örneğin,6 indexte ridge 1.76 katsayısı verdiği feautura,lasso önenli feature olarak görmemiş sıfırlamış 
# Lasso belli featurları sıfırlamış oluyor ridge ise mutlaka bir katsayı veriyor 
# Lasso da multicollinearity varsa. Lasso bi tanesini otomatik atıyor
# 7. index lasso -0.710631 , 8. index -0.000000


# Şimdi best alphayı bulup skorlarımızı yükseltelim
####Choosing best alpha value with Cross-Validation

lasso_cv_model = LassoCV(alphas = alpha_space, cv = 10, max_iter=100000, random_state=42) # will be used in pipeline later
#  max_iter=100000 : gradient descent in attığı adımları yükseltiyoruz yeterli olmadığı için
pipe_lassocv = Pipeline([("scaler", scaler), ("lassocv", lasso_cv_model)]) # pipeline is used to combine scaler and model
pipe_lassocv.fit(X_train, y_train)

pipe_lassocv["lassocv"].alpha_ # Best alpha for lasso 0.1
#0.1

###### Lasso(alpha =0.1) alpha nın 1 olduğu model üzerinden tahmin alıyoruz 

y_pred = pipe_lassocv.predict(X_test)   
y_train_pred = pipe_lassocv.predict(X_train)
lcs = train_val(y_train, y_train_pred, y_test, y_pred, "lasso_cv")
lcs
# CV yapmıştım skorlarımı yorumlayabilirim
#      lasso_cv_train  lasso_cv_test
#R2          0.900491       0.907944
#mae         1.966753       1.766723
#mse         6.295287       4.657759
#rmse        2.509041       2.158184

#Skorlar birbirine yakın overfitting durumu yok

pd.concat([ls,rs, rcs, lss, lcs], axis = 1)
# lassoda 3.760858 dan hata 2.509041 düştü. Çünkü alphayı düşürmüştük. Böylelikle
# Skorlar birbirine yakınlaştı(Train ve test skorları)
# Genel işleyiş olarak da train ve test skorlarının olabildiğince birbirlerine yakın olmasını istiyoruz 

pipe_lassocv["lassocv"].coef_ 
#array([ 6.36058116,  0.        ,  6.44860782, -2.38557078,  0.46733827,
#       -2.09003844, -0.22823288, -1.08177398,  0.        ])

# 0.90 lık skoru 7 feature ile almışız(0.900491)
# Ridge de 9 feature ile 0.90 lık skor mu(0.906476) yoksa lasso da 7 feature ile (0.9) luk skor mu?

lcm_df = pd.DataFrame(pipe_lassocv["lassocv"].coef_, columns = ["lasso_cv_coef_0.1"])
pd.concat([lm_df, rm_df, rcm_df, lsm_df, lcm_df], axis = 1) #Bulduğumuz tüm coeffleri dataframe atıp inceliyoruz
#    lm_coef  ridge_coef_1  ridge_cv_coef_0.91  lasso_coef_1  lasso_cv_coef_0.1
#0  4.280152      5.010922            5.005213      4.821314           6.360581
#1 -2.222880     -1.377652           -1.389454     -0.000000           0.000000
#2  3.932020      4.806274            4.796521      4.880053           6.448608
#3 -4.770839     -3.906449           -3.931707     -0.819769          -2.385571
#4  0.121100      0.261385            0.258058      0.000000           0.467338
#5 -5.277291     -4.066443           -4.092323     -0.351495          -2.090038
#6 -2.611313     -1.746605           -1.763298     -0.000000          -0.228233
#7 -2.188378     -1.954987           -1.969969     -0.710631          -1.081774
#8  1.465706      1.133500            1.155002     -0.000000           0.000000

# (7 fearures ile) test_r2 = 0.90
# Alpha 1 iken çok fazla regularization yapmıştık(4 feature atmıştı)
# lasso_cv_coef_0.1: Daha az bir regularization yaptı ve skorlarımda iyileşme oldu(0.90 a çıktı)




#%%

#########Elastic net


from sklearn.linear_model import ElasticNet, ElasticNetCV
elastic_model = ElasticNet(alpha=1, l1_ratio=0.5, random_state=42) # l1_ratio is used to control the amount of L1 and L2 regularization
# alpha=1 , l1_ratio=0.5 : default
pipe_elastic = Pipeline([("scaler", scaler), ("elastic", elastic_model)]) # pipeline is used to combine scaler and model
pipe_elastic.fit(X_train, y_train) # Pipelinedan sonra modelimi eğitim alpha=1 iken

y_pred = pipe_elastic.predict(X_test)
y_train_pred = pipe_elastic.predict(X_train)   #..tahminleri alıyoruz test ve traain üzerinden
es = train_val(y_train, y_train_pred, y_test, y_pred, "elastic")
es
#      elastic_train  elastic_test
#R2         0.636729      0.635031
#mae        3.802838      3.555946
#mse       22.981720     18.466419
#rmse       4.793925      4.297257

#Skorlar çok düşük,hatalar çok yüksek 

pd.concat([ls,rs, rcs, lss, lcs, es], axis = 1)
# Lasso default 3.760858	3.168025
# Elastic feault 4.793925	4.297257
# Underfitting var diyebiliriz skorlara bakarak

####For Elastic_net CV with Default alpha = 1 and l1_ratio=0.5
model = Pipeline([("scaler", scaler), ("elastic", ElasticNet(alpha=1, l1_ratio=0.5, random_state=42))])

scores = cross_validate(model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=5)
scores = pd.DataFrame(scores, index = range(1, 6))
scores.iloc[:,2:]
#    test_r2  test_neg_mean_absolute_error  test_neg_mean_squared_error  \
#1  0.826285                     -3.154233                   -13.962787   
#2  0.499657                     -3.387816                   -18.345441   
#3  0.676319                     -2.382057                    -8.995287   
#4  0.681643                     -4.372939                   -34.945941   
#5  0.736091                     -3.244457                   -14.906371   

#   test_neg_root_mean_squared_error  
#1                         -3.736681  
#2                         -4.283158  
#3                         -2.999214  
#4                         -5.911509  
#5                         -3.860877  

scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()
# Asıl skorlarımı 0.54 müş
# underfitting açık bir şekilde görünüyor
#test_r2                              0.545808
#test_neg_mean_absolute_error        -4.183244
#test_neg_mean_squared_error        -27.882875
#test_neg_root_mean_squared_error    -5.132246

train_val(y_train, y_train_pred, y_test, y_pred, "elastic") #tek seferlik skorlarımıza tekrar baktık 
#      elastic_train  elastic_test
#R2         0.636729      0.635031
#mae        3.802838      3.555946
#mse       22.981720     18.466419
#rmse       4.793925      4.297257

pipe_elastic["elastic"].coef_ # 2 feature ı kesmiş Elastic model(Lassodan dolayı)
#array([ 2.61657059, -0.73015253,  2.67139945, -1.20195947,  0.        ,
#       -1.16386814, -0.32234008, -0.82569551, -0.        ])
em_df = pd.DataFrame(pipe_elastic["elastic"].coef_, columns=["elastic_coef_(alp:1, L1:0.5)"])
pd.concat([lm_df, rm_df, rcm_df, lsm_df, lcm_df, em_df], axis = 1)#coefleri tekrar inceliyoruz 



######Grid Search for ElasticNet

from sklearn.model_selection import GridSearchCV 
# Elastic net kendine ait Elasticnet cv var ama burada gridsearchcv kullanacağız
elastic_model = ElasticNet(max_iter=10000, random_state=42) 
pipe_elastic = Pipeline([("scaler", scaler), ("elastic", elastic_model)]) # pipeline is used to combine scaler and model
param_grid = {"elastic__alpha":alpha_space,
            "elastic__l1_ratio":[0.1, 0.5, 0.7,0.9, 0.95, 1]}
#alpha_space:0 dan 100 ekadar 100 farklı değer yukarda tanımlamıştık
#l1_ratio(ne kadar ringe ne kadar lasso davransın),1 çıkması durumu Elastik netin tamamen Lasso 
#olarak davrandığını ifade eder.0.5(yarı ridge yarı lasso)
grid_model = GridSearchCV(estimator = pipe_elastic, param_grid = param_grid, scoring = 'neg_root_mean_squared_error',
                         cv =10, verbose =2)
# verbose : çıktıdakileri yazdırması için
grid_model.fit(X_train, y_train)


grid_model.best_params_
# 'elastic__l1_ratio': 1 : direk lassoyu seçti hatalar aynı olacak concat kısmında göreceğiz
#{'elastic__alpha': 0.1, 'elastic__l1_ratio': 1}
y_pred = grid_model.predict(X_test)
y_train_pred = grid_model.predict(X_train)
gm = train_val(y_train, y_train_pred, y_test, y_pred, "elastic_grid")
gm
#      elastic_grid_train  elastic_grid_test
#R2              0.900491           0.907944
#mae             1.966753           1.766723
#mse             6.295287           4.657759
#rmse            2.509041           2.158184

#Lasso ile tamamen aynı çnkü l1_rarito 1 seçilmişti 
pd.concat([ls,rs, rcs, lss, lcs, es, gm], axis = 1)

######Feature importances with Ridge
#yellewbrick kütüpanesi üzzerinden feature importance ı görüntüleyebiliyoruz
ridge_cv_model.alpha_
#0.9181818181818181
from yellowbrick.model_selection import FeatureImportances 

model = Ridge(alpha=pipe_ridgecv["ridgecv"].alpha_)  # ridge_cv_model.alpha_ = 0.91
viz = FeatureImportances(model,labels=list(X.columns),relative=False)
viz.fit(X_train_scaled,y_train)
viz.show()

#Feature importances with Lasso
pipe_lassocv["lassocv"].alpha_
#0.1
from yellowbrick.model_selection import FeatureImportances

model = Lasso(alpha=pipe_lassocv["lassocv"].alpha_)  # lasso_cv_model.alpha_ = 0.1
viz = FeatureImportances(model,labels=list(X.columns),relative=False)
viz.fit(X_train_scaled,y_train)
viz.show()

 










































