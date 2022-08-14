"""LESSON 3 

bıas:Modelin dataları çok fazla genellemesidir.

UNDERFIT             IDEAL                   OVERFIT
Az parametr          low                  çok parametre 
High Bias         low variance              Low bias
Low varyan                                high varyans
data->simple                               data-->complex


UNDERFIT:modelin data eğitimine ihtiyacı var
    *Datayı çok fazla geneller
    *Hem eğitim hemde test datasında büyük hatalar yapar 
    *Bu yüzden varyansı düşük çıkar
    *Model hiçbir şey öğrenmemiş olur

OVERFIT:
    *çok parametre olunca bütün noktalarına girer
    *datayı ezberler
    *train datada varyans --->düşük 
    *test datasında varyans ---<yüksek    çünkü ezberledi


HOW TO RECOGNIZE UNDERFITTING-OVERFITTING ?

underfitting:High training error,high test error
overfitting:low training error,high test error(en çok bununla karşılaşacağız) 

*hem train hem test datasında düşük hatalar alınması beklenir 

Variance:Train ile test datası arasındaki fark 

Comlexity çok düşük olursa model underfit(çok karşılaşmayız),çok yüksek olursa 
overfit olur.Biz ne çok ne de çok düşük olsun istiyoruz 

Bias: train setindeki gerçek değerle tahmin edilen değer arasındaki fark

Underfitting ve Overfitting ile mücadele için;
underfittig-->feature eklenir(Comlexity artar)
Overfitting-->Feature azaltılır(Comlexity düşer)
datanın azlığından kaynaklanıyorsa data arttırılır yada derece düşürülür

*overfitting de daha iyi bir sonuç alır mıyım diye cross validation yapılabilir
*yada regularization(lassa-ridge)yapılabilir
 
                    Underfitting          Overfitting
Bias                   high                   low
Variance               low                    high
Complexity             low                    high
Flexibillity           low                    high
Generalizability       high                   low

Types of regression models 
*Simple linear regression   y=b0+b1x
*Multi linear regression 
*Polynomial Regression      y=b0+b1x+b2x^2...

Smple Linear-->1 feature, 1 label 
Multi linear-->birden fazla feature,1 label
Polynomial-->Multinin özel biçimi
*Non-linear datalarda iyi çalışır 
*Polynomial degresini seçmek çok önemli 
*degree=1-->Linear, degree=düşük-->underfit ,degree=yüksek-->overfit 

*Polynomial regressionda Bias Variance Trade-off(Bias-Variance degresi)ni 
ayarlaMak çok önemli 

Regularization --->overfitting ve Multicallinearity ile ücadele 
regularization-->*performansı yüksektir 
*trainde testde makul sonuçlar verir 

*Multicallinearity, Linear modellerde yaşanan bir sorundur. 
*Advance modellerde bunların arkada çalışan paraetreleri var ama linear 
modellerde yok 

***"""


#NOTEBOOK

# Lineer Regression ın devamı. Polynomlarla bu regressipn ı çözmeye çalışacağız
# Underfitting ve Overfitting göreceğiz bu gün

# ÖNCEKİ DERSIN ÖZETİ
# Residuals
    # Sum and mean of the Residuals are always Zero
    # Residuals are normally distributed for suitable Linear Regression
# Regression Error Metrics
    # MAE, MSE, RMSE
# Scikit-learn Library and ML
    # 5 steps: import, split data, model building and fit, prediction, evaluation

# Konular
    # Introduction to Bias-Variance Trade-off
    # Underfitting and Overfitting Problems
    # Training Errors vs Validation Error(Test Error)
    # Polynomial Regression

# Introduction to Bias-Variance Trade-off
# varyans hataların dağlımı demek 
# Bias, bir eşik/aralık şeklinde tanımlayabiliriz. Varyans demek değişim demek
# .. Heterojen yapıda değişim çok fazla olur. Homojen yapıda değişim çok az olur
# .. Bias, Data Scienceda da bu residual e karşılık geliyor.
# 3 kavrama bakacağız
    # Underfitted      : Models with too few parameters may not fit the data well(high bias)
        # .. but are consisten accross different training sets(low variance)
        # Residual erin ya da bias ın burada çok yüksek olduğunu söyleriz, varyans düşük(dümdüz doğru)
        # -- Simple Model Underfit
        # Bu durumda model veriyi iyi öğrenememiş. Simple bir model ortaya çıkarmış complex bir data için
        # Buna underfitting denir
        # Yeterli feature olmadığında ya da model complexity düşük olduğunda kaynaklanır
        # Veriyi arttırmakla, Ya da complexity yi arttırmakla çözülebilir
    # Good Fit/Robust  : bias ve variance ideal değerlerde(Low bias & low variance)
    # Overfitted       : Models with too many parameters may fit the training data well(low bias)
        # but are sensitive to choice of training set(high variance) -- Complex Model-- OverFIT
        # Burada varyans çok yüksek, bias çok düşük
        # Bu durumda model veriyi ezberlemiş. Hata 0 a yakın. Training sonucu %100 çıkar ama
        # .. test(görmediği) datası geldiğinde başarısının düşük olduğunu görürüz
        # NOT: Bundan dolayı train hatası ile test hatası birbirine yakın olmalı
        # Model complex olmasından kaynaklanır. Çözmek için model complexity düşürülebilir
        # Parametre azaltılması, More training data/ Cross/validation ile, Regularization(Lasso&Ridge) gibi şeylerle çözülür
    # Orion hoca: 
        # low bias low variance(underfittig)   modelin trainden aldığı scorelar ile testten aldığı scoreların ikisininde  çok düşük olması ile karşımıza çıkar.
        # low bias high varyans(overfitting) ise model train datası üzerinde çok çok iyi score alırken test datasına gelince kötü scorelar olarak karşımıza çıkar.

# Model complexity
    # Train ve test hataları ile alakalı bir konu Train hatası ile test hatasının belli bir oranda
    # .. olması gerekir. Grafiklere bakmak daha anlamlı burada. Ayrıca altta da bahsedilecek
    # Orion Hoca : bizim istediğimiz model train datası üzerinden bir genelleme yapsın bazı hataları olsun bunun karşılığında da test datası üzerinde hata makul seviyede kalsın.


# Kurs boyunca 2 anayol var: Regression ve Classification
# Regression bir değer buluyoruz(Salary, Amount vs)
# Classification da sınıflandırma yapıyoruz(0 mı 1 mi, sıcak mı soğuk mu, Hayatta mı değil mi ... gibi)
# Alttaki classification da grafikte sağ üst underfitting. Sağ alt overfitting


#Polynomial Regression¶
#Polynomial Regression is a form of regression analysis in which the relationship between the independent variables and dependent variables are modeled in the nth degree polynomial.

#Polinom Regresyon , bağımsız değişkenler ile bağımlı değişkenler arasındaki ilişkinin n'inci derece polinomda modellendiği bir regresyon analizi şeklidir.

#Types of polinomials

#1st degree ---> linear b1x + b0

#2nd degree ---> Quadratic b2x**2 + b1x + b0

#3rd degree ---> Cubic b3x3 + b2x2 + b1x + b0 (third order equation)

#interview sorusu:Bir datanın lineer regresyona uygun olup olmadığını nasıl anlıyorsunuz?
#*target datasındaki gerçek değer ve tahmin değerlerinin residual larının 
#normal dağılıma uyup uymadığını kontrol ediyoruz.
#*bunlar mutlaka random dağılacak 

#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)

#Polynomial Features
from sklearn.preprocessing import PolynomialFeatures

# içerisine datayı verdiğimiz zaman verdiğimiz dereceye göre feature sayısını arttırıyor
data = np.array([[2, 3, 4]])   # Çalışma mantığını anlatmak için array ürettik
print(data)

trans=PolynomialFeatures(degree=2,include_bias=False)# Datayı polinomial Featurelara dönüştürüyoruz
# include_bias = False : İlerde neden False olduğundan bahsedilecek(basitçe formülde b0 ın olmaması) 

trans.fit(data) # 2, 3, 4 , 2x3, 2x4, 3x4, 2**2, 3**2, 4**2  # Veriyi öğrenme/tanıma aşaması

# NOT: Orion hoca: fit kalıbını çıkar, transform o kalıbı uygula demek
# 2, 3, 4, 2x3, 2x4, 3x4, 2**2, 3**2, 4**2, 2x3x4, 3x2**2, 4x2**2, 2x3**2, 4x3**2, 2x4**2, 3x4**2, 2**3, 3**3, 4**3

trans.transform(data)  # Veriyi fit ettikten sonra dönüştürüyor. Ancak alttaki komut tek aşamada yapıyor
#array([[ 2.,  3.,  4.,  4.,  6.,  8.,  9., 12., 16.]])

trans.fit_transform(data) # combining method # Daha kullanışlı
# Orion hoca: Eldeki feature lar ile sentetik feature lar ürettik diyebiliriz
#array([[ 2.,  3.,  4.,  4.,  6.,  8.,  9., 12., 16.]])

df = pd.read_csv("C:\\Users\\hp\\Desktop\\DataScience\\Machine_Learning\\SofiaHoca\\Machine_Learning-main\\Machine_Learning-main\\In Class\\Lesson 1\\Advertising.csv")
df.head()

#      TV  radio  newspaper  sales
#0  230.1   37.8       69.2   22.1
#1   44.5   39.3       45.1   10.4
#2   17.2   45.9       69.3    9.3
#3  151.5   41.3       58.5   18.5
#4  180.8   10.8       58.4   12.9


#Polynomial Converter

X=df.drop(columns="sales",axis=1)# sales haricindeki diğer değişkenleri seçiyoruz(Independent variables)
y=df.sales

polynomial_converter=PolynomialFeatures(degree=2,include_bias=False)
polynomial_converter.fit(X)

poly_features=polynomial_converter.transform(X)
poly_features


#array([[ 230.1 ,   37.8 ,   69.2 , ..., 1428.84, 2615.76, 4788.64],
#       [  44.5 ,   39.3 ,   45.1 , ..., 1544.49, 1772.43, 2034.01],
#       [  17.2 ,   45.9 ,   69.3 , ..., 2106.81, 3180.87, 4802.49],
#       ...,
#       [ 177.  ,    9.3 ,    6.4 , ...,   86.49,   59.52,   40.96],
#       [ 283.6 ,   42.  ,   66.2 , ..., 1764.  , 2780.4 , 4382.44],
#       [ 232.1 ,    8.6 ,    8.7 , ...,   73.96,   74.82,   75.69]])

#polynomial_converter.fit_transform(X)

poly_features.shape # Shape de columns 3 dü, 9 a çıktı(2.dereceden olduğu için)
#(200, 9)


Train | Test Split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(poly_features,y,test_size=0.3,random_state=101)
# test_size    : train datasının ve test datasının oranını belirleme. Test oranı:%30, Train oranı:%70
# random_state : Her seferinde aynı sonuçları almak için kullanılır(Çalışırken hocayla aynı sonuçları almak için)

X_train.shape
#(140, 9)
from sklearn.linear_model import LinearRegression
model_poly=LinearRegression()
model_poly.fit(X_train,y_train)

my_dict = {"Actual": y_test, "pred": y_pred, "residual": y_test-y_pred}
compare = pd.DataFrame(my_dict)
compare.head(5)
#     Actual       pred  residual
#37     14.7  13.948562  0.751438
#109    19.8  19.334803  0.465197
#31     11.9  12.319282 -0.419282
#89     16.7  16.762863 -0.062863
#66      9.5   7.902109  1.597891

#tahminler iyi gözüküyor 

compare.head(20).plot(kind='bar',figsize=(15,9))
plt.show();


#Poly Coefficients


model_poly.coef_
#array([ 5.17095811e-02,  1.30848864e-02,  1.20000085e-02, -1.10892474e-04,
#        1.14212673e-03, -5.24100082e-05,  3.34919737e-05,  1.46380310e-04,
#       -3.04715806e-05])


df_coef = pd.DataFrame(model_poly.coef_, index = ["TV", "radio", "newspaper", "TV^2", "TV&Radio", \
                                   "TV&Newspaper", "Radio^2", "Radio&newspaper", "Newspaper^2"], columns = ["coef"])

df_coef 
# John Hoca: Standartlaştırma yapmadığımız için alttaki feature ların önemlerini sıralamak çok doğru değil.
# .. Bunu Ridge ve Lasso da detaylı göreceğiz

#                     coef
#TV               0.051710
#radio            0.013085
#newspaper        0.012000
#TV^2            -0.000111
#TV&Radio         0.001142
#TV&Newspaper    -0.000052
#Radio^2          0.000033
#Radio&newspaper  0.000146
#Newspaper^2     -0.000030


model_poly.predict([[2.301000e+02, 3.780000e+01, 6.920000e+01, 5.294601e+04,
       8.697780e+03, 1.592292e+04, 1.428840e+03, 2.615760e+03,
       4.788640e+03]])

array([21.86190699])

#Evaluation on the Test Set

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

y_pred = model_poly.predict(X_test) # Tahminlerimizi yaptık.(Dikkat: X_test ile)
# Şimdi metriklerimize bakarak tahminlerin ne kadar iyi olduğuna bakalım


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("Model performance:")
    print("--------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")


eval_metric(y_test, y_pred)

#Model performance:
#--------------------------
#R2_score 	: 0.9843529333146795
#MAE 		: 0.48967980448035575
#MSE 		: 0.44175055104033906
#RMSE 		: 0.6646431757269001

y_train_pred = model_poly.predict(X_train) # Tahminlerimizi yaptık.(Dikkat: X_train ile)

eval_metric(y_train, y_train_pred) 
# John Hoca: X_test ile X_train metric leri arasında çok fark olmadığı için
# .. underfitting, overfitting durumu yok diyebiliriz

#Model performance:
#--------------------------
#R2_score 	: 0.9868638137712757
#MAE 		: 0.4049248139151643
#MSE 		: 0.34569391424439977
#RMSE 		: 0.5879574085292231

#Simple Linear Regression:
#MAE : 1.213
#RMSE : 1.516
#r2_score : 0.8609

#Polynomial 2-degree:
#MAE : 0.48
#RMSE : 0.66
#r2_score : 0.9868

# Şu anda 2. dereceden bir polynomun(lineer regression a göre) daha iyi sonuçlar ürettiğini söyleyebiliriz
# Acaba derece artarsa daha mı iyi olur sonuçlar. Buna bakacağız altta
# NOT: Polynomial regression un dezavantajının bu olduğu söylenmişti

#Let's find optimal degree of poly

def poly(d):
    
    train_rmse_errors = []
    test_rmse_errors = []
    number_of_features = []
    degrees=[]
    
    for i in range(1, d):
        polynomial_converter = PolynomialFeatures(degree = i, include_bias =False)
        poly_features = polynomial_converter.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=101)
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_RMSE = np.sqrt(mean_squared_error(y_train,train_pred))
        test_RMSE = np.sqrt(mean_squared_error(y_test,test_pred))
        
        train_rmse_errors.append(train_RMSE)
        test_rmse_errors.append(test_RMSE)
        
        number_of_features.append(poly_features.shape[1])
        degrees.append(i)
        
    return pd.DataFrame({"train_rmse_errors": train_rmse_errors, "test_rmse_errors":test_rmse_errors, "Degree":degrees}, 
                        index=range(1,d))



poly(10) # Test hatası belli bir seviye sonra veriyi ezberlemiş olduğunu görüyoruz
# .. Çünkü veriyi o kadar ezberlemiş ki train hatasında hata neredeyse hiç yok ama
# .. hiç görmediği veri ile(test verisi) karşılaştırınca ezberlediği için tahminler doğru gelmemiş


#   train_rmse_errors  test_rmse_errors  Degree
#1           1.734594          1.516152       1
#2           0.587957          0.664643       2
#3           0.433934          0.580329       3
#4           0.351708          0.507774       4
#5           0.250934          2.575837       5
#6           0.194567          4.214027       6
#7           5.423737       1374.957405       7
#8           0.141681       4344.727851       8
#9           0.170935      93796.026718       9

#5 ten itibaren overfitting başladı(train seti ile test seti arasında fark açıla durumu )

# Derece 1 den 9 a kadar
plt.plot(range(1,10), poly(10)["train_rmse_errors"], label = "TRAIN")
plt.plot(range(1,10), poly(10)["test_rmse_errors"], label = "TEST")
plt.xlabel("Polynamial Complex")
plt.ylabel("RMSE")
plt.legend();



# Derece 1 den 5 e kadar olan kısmı inceleyelim
# Derece 4-5 arası ezberleme yapmış
# derece 1 de de underfitting var
plt.plot(range(1,6), poly(6)["train_rmse_errors"], label = "TRAIN")
plt.plot(range(1,6), poly(6)["test_rmse_errors"], label = "TEST")
plt.xlabel("Polynamial Complex")
plt.ylabel("RMSE")
plt.legend();

# Hangi noktayı seçeceğiz optimal değeri seçmek için
# Orion Hoca: Sınır değerlerden(2.0 ve 4.0) uzak durulmalı(underfitting ve overfitting e gitmeye meyilli)
# .. O yüzden orta yolu seçmek daha anlamlı(yani 3.0)

#Finalizing Model Choice

final_poly_converter = PolynomialFeatures(degree = 3, include_bias=False)

final_model = LinearRegression()

final_model.fit(final_poly_converter.fit_transform(X), y) 
# Burada datayı bölmedik artık çünkü optimal noktayı belirlediğimiz için tüm datayla eğitimi yapalım ki 
# .. en ideal tahminleri elde edelim


#Predictions
new_record = [[150, 20, 15]]

new_record_poly = final_poly_converter.fit_transform(new_record) 
# Burada new_record_poly formatını değiştiriyorum

new_record_poly

#array([[1.500e+02, 2.000e+01, 1.500e+01, 2.250e+04, 3.000e+03, 2.250e+03,
#        4.000e+02, 3.000e+02, 2.250e+02, 3.375e+06, 4.500e+05, 3.375e+05,
#        6.000e+04, 4.500e+04, 3.375e+04, 8.000e+03, 6.000e+03, 4.500e+03,
#        3.375e+03]])

final_model.predict(new_record_poly) # Sonuç: 14.24
# Kulanıcı derse ki eğer benim tv değerim 150, radio :20, newspaper:20 olursa sonucum(sales) ne olur?? --> 14.24

#array([14.24950844])

#Overfitting

# Dereceyi 5 seçerek overfitting durumunu gözlemleyelim
over_poly_converter = PolynomialFeatures(degree =5, include_bias =False)

over_model=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(over_poly_converter.fit_transform(X), y, test_size=0.3, random_state=101)
over_model.fit(X_train, y_train)

y_pred_over = over_model.predict(X_test)
eval_metric(y_test, y_pred_over)


#Model performance:
#--------------------------
#R2_score 	: 0.7649916528404768
#MAE 		: 0.6659637641421313
#MSE 		: 6.634794172264552
#RMSE 		: 2.575809420796607

y_train_over = over_model.predict(X_train)
eval_metric(y_train, y_train_over)
# Gördüğümüz gibi metricler arasındaki değerlerde kayda değer(örneğin R_2 0.99-0.76) fark var
# RMSE ye bakarsak train datasında hata çok çok düşük(0.25) ama hiç görmediği test te 2.575 e çıkmış

#Model performance:
#--------------------------
#R2_score 	: 0.9976072484167179
#MAE 		: 0.1862092141111129
#MSE 		: 0.06296802178630591
#RMSE 		: 0.2509342977480478



#Underfitting

#Testing data performance:
#--------------------------
#R2_score 	: 0.8609466508230367
#MAE 		: 1.5116692224549084
#MSE 		: 3.796797236715222
#RMSE 		: 1.9485372043446392


#Training data performance:
#--------------------------
#R2_score 	: 0.9055159502227753
#MAE 		: 1.1581502948072524
#MSE 		: 2.4793551789057866
#RMSE 		: 1.574596830590544


# Gördüğümüz gibi metricler arasındaki değerlerde fark var
# r2 de performance düşmüş(0.90 dan, 0.86 ya). Çok düşmemiş ama bu data için ideal
# .. olan değerler daha yüksek olmalıydı
# RMSE de de fark var

# NOT: Ares hoca: R2 score negatif olması aşırı ezberi ve best fit line nın ters çizildiğini gösterir

# NOT: Orion hoca: fit kalıbını çıkar. Transform o kalıbı uygula

