
----------------------------------------------------------------------------
INCLASS 1

RULES       Traditional    ---->answers
DATA        Programing

burada kuralı ne koyarsanız çıktıyı o verir 
örk:data=2 ve 3 rules=add  answers=5 olur 


ANSWERS      Machine    ------------>rules 
DATA         Learning 

Algoritma üzerinden başka bir algoritma üretir
örk: answers:9 7 8 data:6 3 - 4 2 - 4 4  data ve aswers a bakarak add  rule unu bulur 


What is Machine Learning?
*Recommendation engines
*Customer Churn
*New Pricing Models 
*Email Spam Filtering 
*Material and Stock Estimates 
*Pattern and Image Recognıtion
*Predictive iventory Plan
*Purchasing Trends 
*Credit Scoring 


independent Variables(bağımsız değişken "X")Features
Dependent Variables(bağımlı değişken "y")Target-Label 

test:test için seçilecek satırlar(gözlem)
train:eğitim için seçilecek satırlar 
train ve test datası randomly ayrılır 


   supervised learning overview
supervised learning(denetimli öğrenme):data başka bir machine learning algoritması 
tarafından yada kişiler tarafınfan label lanmış.
feature ları kullanarak algoritmayı çalıştırır ve ortaya bir rule koyar 
rule data gibi bir satır verildiğinde  o datanın ne olduğu çıktısını verir 

*it is the process of learning from labeled observations 
*labels teach the algorithm how to label the observations 


   supervised learning overview

unsupervised learning(denetimsiz öğrenme):
*label yoktur verilen datayı çıktı olarak kümelere ayırır 
*in unsupervised learning the machine learns from unlabeled data 
*there is no training data for unsupervised learning 


Correlation:İki nicel değişken arasındaki ilişkinin gücünü verir.
r ile gösterilir.-1 ile +1 arasında gösterilir 

güçlü      ilişki yok       güçlü  
 -1           0               1

Correlation yoksa lineer regresyonda olmaz 
  
*the correlation summarizes the direction of the association between 
two quantitative variables and the strenght of its linear trend  


Regression:Bit feature da bi değişiklik olurken diğer feature ın 
nasıl etkilendiğine bakar


Correlation(r)
*iki veri arasındaki ilişkiye bakar 
*variable lar birlikte hareket eder(scatterplot ortaya çıkar)
*data -1 ile +1 arasında bir rakamla ifade edilir
*is there a relationship between X and y

Regression 
*Bir feature ın diğer feature ı nasıl etkilediğine bakar 
*Cause and effect üzerine çalışır(etki-tepki)
*data bir line ile ifade edilir 
*What is the relationship between X and y


Linear Regression:istatiksel regresyon modellerini kullanarak tahmin analizi yapar
*statistical regression method used for predictive analysis 
*shows the relationship between the independent varianle and the dependet variable 
*contiunes variable üzerinden çalışır.


Linear Regression Theory
 Y=bı+b1X (regression equation)
 
ei=Yi-^Yi  random error 
bi=slope eğri eğimi  
b0=intercept x in sıfır olduğu yerde y nin alacağı değer 
*amaç bütün noktalara en yakın geçecek çizgiyi bulmak

method of least squares:
Σei^2=Σ(y−yˆ)^2


Gradient Descent:
*Gradient descent is an algorithm that finds best fit line for given training 
dataset
*Error leri 0 a yaklaştırmaya çalışır.En küçük olduğu yerde durur ve line ı çizer 


The Coefficient of Determination (R^2):
*Elimizdeki data ile tahmin etmek istediğimizin ne kadarını karşılıyoruz 
0-hiç tanımlamaz 
1-çok iyi tanımlar 

*modelimiz iyi mi kötü mü ? bunun için kullanılır 

R^2=1-(Σ(yi−yiˆ)^2)/(Σ(yi−y)^2)

yiˆ :y lerin tahmini 
y   :y lerin ortalaması 


simple linear regression: Y=bı+b1X
multiple linear regression: y=bo+b1*x1+b2*x2+.....+bn*xn
y->dependent variable 
x1..Xn -->independent variables 


Regression Error Method:

1)Mean Absulate Error(MAE):
    1/nΣ =|yi-yi^|
dezavantajı:Outlier hatalarını minimize etmez(cezalandırmaz)

2)Mean Squared Error(MSE):Hatanın varyansı 
    
 1/nΣ(yi-yi^)^2
 
*hataların karesi alındığı için cezalandırması iyi 
dezavantajı:açıklamsı zor 

3)Root Mean Squared Error(RMSE):Hatanın standart sapması 

√(1/nΣ(yi-yi^)^2)

*hem hatayı cezalandırır hem açıklaması kolaydır 

 Scikit-Learn Kütüphanesi:

*Supervised ve unsupervised Learning i destekler. 
-model fitting 
-data preprocessing 
-model selection 
-evaluation  
aşamalarında kullanılır 

 ML Aşamaları 
 
1)import aşamaları 

2)Splitting Data (datayı bölme)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=30,random_state=42)

3)Fitting Data(Model Kurma)

model.fit(X_train,y_train)
logmodel=LogisticRegression() ---->önce model oluşturulur 
logmodel.fit(X_train,y_train)---->sonra bu modelin içine X_train,y_train atılır

4)Predicting Data(Fit te öğrendiklerine göre tahmin yapar)
    predictions=model.predict(X_test)
    
5)Probability of Predicted Data 
    model.predict_proba(X_test)
*bu adım ,classification problemlerinde olasılık istendiğinde kullanılır 

6)Evaluation(Değerlendirme)
*MAE,MES,RMSE kullanılarak evaluate işlemi yapılır 


***********NOTEBOOK*****************
# Elimizdeki data çok büyükse Deep Learning ile çalışılır
# ML de structured datalarla çalışılır. DL de farklı datalarla da çalışılır
# Correlation & Linearity
# Korelasyon : Iki sayısal değişken arasındaki ilişki(Yönü(pozitif ve negatif) ve kuvveti(weak and strong) var)
# .. -1, +1 arasında değerler alır. Simgesi r dir
# Linearity  : iki feature arasında lineer bir ilişkiden bahsetmek için bunu bir doğru
# .. üzerinde ifade edebilmemiz lazım.(y=ax+b)
# .. Lineer ilişki
# r =  1 kuvvetli ve pozitif ilişki
# r = -1 kuvvetli ve negatif
# r = 0.6 ilişki var ama çok değil
# r = -0.4 ilişki var ama düşük
# r = 0 ilişki yok

# correlation ilişki ve variables move together and data represent in single point,
# regression bir değişkenin diğer değişkeni ne kadar etkilediği, neden ve etki var regresyon da, data represent by line
#Simple Linear Regression - Supervised Model
# Parametrik algoritmadır. (Matematiksel bir denklem ile ifade edilen)
# Continuous variable lar tahmin edilmeye çalışılır
# Simple ve Multi olarak ikiye ayrılır
# b0: intercept        : y eksenini kestiği nokta
# b1: slope            : Eğim
# ei = Yi - y(head)i   : Hata
# Y(head) = b0 + b1*X  : Regression Equation
# Salary = b0 + b1*Experience

# method of Least Squares : TOPLAM(ei(kare)) = TOPLAM(Yi-Y(head)i)(kare)
# NOT: Best fit line ı hata kareler toplamını minimize ederek çizeriz. Yani
# .. least square metodu kullanılır algoritma olarak gradient descent i kullanırız
# NOT: Bütün hataların toplamı best fit line için sıfır olur. Bu yüzden kareler alınır
# NOT: gradient descent kullanırken karelerin türevini almak daha kolay olduğundan karalaeri almak daha kullanışlıdır

# Gradient Descent: Rasgele line lar çizer hataları hesaplar. Başka line çizer tekrar hesaplar
# .. böyle devam ediyor. En son hata bir yerde artıyor. O anda duruyor ve bir önceki benim
# .. best fit line ım diyor

# R(kare):Bağımsız değişkenlerin bağımlı değişkeni açıklama oranı.
# .. Orion hoca : Elimizdeki featurelar ile targetın varyansını ne kadar açıkladığını gösterir
# .. 0 ile 1 arasında değişir.
# .. Tv Ads --> Cars örneği
"""
week   Number of Tv ads    Number of cars sold
1       3                   13
2       6                   31
3       4                   19
4       5                   27
5       6                   23
6       3                   19
R(kare) = 0.7
# Tv reklamları araba satışlarını %70 oranında açıklıyor(Açıklanabilir varyans %70)
# Kalan %30
"""

# Simple Linear Regression   : y = b0+b1*x1
# Multiple Linear Regression : y = b0+ b1*x1 + b2*x2 + ... + bn*xn


#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')



#Read Dataset

df = pd.read_csv("C:\\Users\\hp\\Desktop\\DataScience\\Machine_Learning\\SofiaHoca\\Machine_Learning-main\\Machine_Learning-main\\In Class\\Lesson 1\\Advertising.csv")
df

#independent variable = feature = estimator = attribute = input
#dependent variable = target = label = output
#rows = observation, sample
#features = TV + radio + newspaper
#target = sales

df.info()

 #   Column     Non-Null Count  Dtype  
#---  ------     --------------  -----  
 #0   TV         200 non-null    float64
 #1   radio      200 non-null    float64
 #2   newspaper  200 non-null    float64
 #3   sales      200 non-null    float64
#dtypes: float64(4)
#memory usage: 6.4 KB

df.describe().T
#           count      mean        std  min     25%     50%      75%    max
#TV         200.0  147.0425  85.854236  0.7  74.375  149.75  218.825  296.4
#radio      200.0   23.2640  14.846809  0.0   9.975   22.90   36.525   49.6
#newspaper  200.0   30.5540  21.778621  0.3  12.750   25.75   45.100  114.0
#sales      200.0   14.0225   5.217457  1.6  10.375   12.90   17.400   27.0

#mean ve std degerleri birbirine yakinsa yada std mean den buyukse outlier olabilir 
#min ile 0.25 lik deger ve 0.75 ve max arasinda ciddi bir fark varsaoutlier olabilir


#Create new independent variable (feature)

# Simple lineer regression ı göstermek için feature engineering yapalım

#modelden en iyi sonuc alinabilmesi icin eldeki feature larla yeni feature olusturldu.(feature engineering)
#tatal_spend in corr u incelenecek,corr en iyi olan fuature alinacak(simple lineer regression icin(tek feature))
df["total_spend"] = df["TV"] + df["radio"] + df["newspaper"]

df

#        TV  radio  newspaper  sales  total_spend
#0    230.1   37.8       69.2   22.1        337.1
#1     44.5   39.3       45.1   10.4        128.9
#2     17.2   45.9       69.3    9.3        132.4
#3    151.5   41.3       58.5   18.5        251.3
#4    180.8   10.8       58.4   12.9        250.0
#..     ...    ...        ...    ...          ...
#195   38.2    3.7       13.8    7.6         55.7
#196   94.2    4.9        8.1    9.7        107.2
#197  177.0    9.3        6.4   12.8        192.7
#198  283.6   42.0       66.2   25.5        391.8
#199  232.1    8.6        8.7   13.4        249.4


#okunmasi kolay olmasi icin tarket sona alindi.bu bir kural degil 
df = df.iloc[:,[0,1,2,4,3]]
df.head()
#      TV  radio  newspaper  total_spend  sales
#0  230.1   37.8       69.2        337.1   22.1
#1   44.5   39.3       45.1        128.9   10.4
#2   17.2   45.9       69.3        132.4    9.3
#3  151.5   41.3       58.5        251.3   18.5
#4  180.8   10.8       58.4        250.0   12.9

sns.pairplot(df)
#en yuksek corr un total_spend ile target(sales) arasinda oldugunu gorduk

#Which feature is more suitable for linear regression?

# We will check correlation for answer.

for i in df.drop(columns ="sales"):
    print(f"corr between sales and {i:<12}:  {df.sales.corr(df[i])}")

#corr between sales and TV          :  0.7822244248616061
#corr between sales and radio       :  0.5762225745710551
#corr between sales and newspaper   :  0.22829902637616528
#corr between sales and total_spend :  0.8677123027017427

sns.heatmap(df.corr(), annot=True)

#corr en yuksek total_spend feature oldugu icin onla devem edilecek
#independent variable:tota_spend ,dependent variable=sales
#simple lineer regression oldugu icin tek feature la devam edildi 
df = df[["total_spend", "sales"]]
df.head()


#Plotting the relationship between independent variable and dependent variable

sns.scatterplot(x ="total_spend", y = "sales", data=df)
#pozitif yonlu kuvvetli iliski
#Correlation between independent variable and dependent variable

corr = df["sales"].corr(df["total_spend"])
corr
#0.8677123027017427
df["total_spend"].corr(df["sales"])
#0.8677123027017427


#Coefficient of determination (R^2)

R2_score = corr**2  # Sadece simple linear regression özelinde olan bir formül. Multi de böyle değil
R2_score  # Feature ekleyerek R^2 yükseltilebilir(Eğer data lineer bir özellik gösteriyorsa)
#0.7529246402599608   elimizdeki ver target i tahmin edebilme icin yuzde 75 oraninda yeterli
#Bağımsız değişkenlerin bağımlı değişkeni açıklama oranı
#target label i ni dogru tahmin edebilmek icin elimde yeterli verinin ne kadari var
#Linear Regression

sns.regplot(x="total_spend", y="sales", data=df, ci=None) #eldeki data icin best fit i cizer
#best fit in altinda ve ustunde olan residual degerleri topladigimizda her zaman sifi cikar 

#Splitting the dataset into X(independent variables) and y (dependent variable)

# y_pred = b1X + b0   b1:slope   b0:intercept(x  0 iken y nin aldigi deger)

X= df["total_spend"]
y= df["sales"]


#Determination of coefficients (slope and intercept)
np.polyfit(X, y, deg=1) #bagimli ve bagimsiz degiskenler ve derecesi verildiginde katsayilari dondurur 
#array([0.04868788, 4.24302822])
slope, intercept = np.polyfit(X, y, deg=1)
print("slope    :", slope)
print("intercept:", intercept)
#slope    : 0.048687879319048145
#intercept: 4.2430282160363255

#Why do we use the least squares error method to find the regression line that best fits the data?

b1, b0 = np.polyfit(X, y, deg=1) # degree=1, linear (doğrusal) model

print("b1 :", b1)
print("b0 :", b0)
#b1 : 0.048687879319048145
#b0 : 4.2430282160363255
y_pred = b1*X + b0

values = {"actual": y, "predicted": y_pred, "residual":y-y_pred, "LSE": (y-y_pred)**2}
df_2 = pd.DataFrame(values)
df_2
#     actual  predicted  residual        LSE
#0      22.1  20.655712  1.444288   2.085967
#1      10.4  10.518896 -0.118896   0.014136
#2       9.3  10.689303 -1.389303   1.930164
#3      18.5  16.478292  2.021708   4.087302
#4      12.9  16.414998 -3.514998  12.355211
df_2.residual.sum().round()   #hatalarin(residual) toplami sifir geldi 

#Prediction with simple linear regression

potential_spend = np.linspace(0, 500, 100)
potential_spend

predicted_sales_lin = b1* potential_spend + b0
predicted_sales_lin

sns.regplot(x="total_spend", y="sales", data=df, ci=None)  
plt.plot(potential_spend, predicted_sales_lin)


# NOT: Modeli eğittiğiniz range önemli 1-5 odalı evler için fiyat tahmini yapıyorsanız,
# .. 8 odalıyı tahmin ederken büyük ihtimalle yanlış sonuç alırsınız
