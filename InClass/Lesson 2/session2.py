#INCLASS 2





# Not: Lineer Regression 1. Assumption: Güçlü bir lineer ilişkisi olması 

# There are four assumptions associated with a linear regression model:
# 1.Linearity: The relationship between X and the mean of Y is linear.
# 2.Homoscedasticity: The variance of residual is the same for any value of X.
# 3.Independence: Observations are independent of each other.
# 4.Normality: For any fixed value of X, Y is normally distributed.


# Residual: Hata = ei = yi-y(head)i
# Not: Hataların toplamı sıfırdır
# 2. Assumption : Hatalar normal dağılım sergilerse bu data lineer regression a uygundur
# Hatalar origin etrafında homojen dağılmalı

# Regression Error Metrics
    # 1.Mean absolute error    : MAE : Ares hoca: Hataları çok cezalandırmıyor
    # 2.Mean Square Error      : MSE : Karesini aldığımızdan dolayı hataları fazla cezalandırıyor.
        # .. o yüzden bunu yorumlamakta da zorlanıyoruz.
    # 3.Root Mean Square Error : RMSE : Büyük hataları büyük, küçük hataları küçük cezalandırıyor.
        # .. o yüzden bunu yorumlamak daha kolay diyebiliriz.
# Genelde RMSE yi kullanacağız. Orion Hoca: Hepsini kullanıp duruma göre değerlendirme yapacağız ancak genelde
# MAE küçük, RMSE ve MSE büyük gelir. Önemli olan bunları yorumlamak
# Orion Hoca: Modeliniz de outlierlar var ise linear regresyon için MAE kötü bir seçim olabilir

# Scikit-Learn Library
# Model fitting, data processing, model selection, evalution vs gibi şeyler yapılabilir

# Machine Learning with Python
    # 1.EDA
    # 2.Splitting Data
        # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
    # 3.Model Building
        # model.fit(X_train, y_train)
            # logmodel= LogisticRegression()
            # logmodel.fit(X_train, y_train)
    # 4.Predicting Data
        # prediction = model.predict(X_test)
            # prediction = logmodel.predict(X_test)
    # 5.Evaluation of the Model
        # R2_score = r2_score(actual,pred)
        
# Orion Hoca : Model oluşturma süreci basitçe :
# 1.EDA
# 2.Train Test Split
# 3.Preprocess (Scale,onehot encodin vb.)
# 4.model building (Linear regresion .......vb)
# 5.model evaluation (erros metricleri üzerinden modelin değerlendirilmesi)



NOTEBOOK


#Multiple Linear Regression and Regression Error Metrics


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\hp\\Desktop\\DataScience\\Machine_Learning\\SofiaHoca\\Machine_Learning-main\\Machine_Learning-main\\In Class\\Lesson 1\\Advertising.csv")
df
#        TV  radio  newspaper  sales
#0    230.1   37.8       69.2   22.1
#1     44.5   39.3       45.1   10.4
#2     17.2   45.9       69.3    9.3
#3    151.5   41.3       58.5   18.5
#4    180.8   10.8       58.4   12.9


df.shape
#(200, 4)

df.info()
 #   Column     Non-Null Count  Dtype  
#---  ------     --------------  -----  
# 0   TV         200 non-null    float64
# 1   radio      200 non-null    float64
# 2   newspaper  200 non-null    float64
# 3   sales      200 non-null    float64
#dtypes: float64(4)

sns.pairplot(df)

df.corr()

#                 TV     radio  newspaper     sales
#TV         1.000000  0.054809   0.056648  0.782224
#radio      0.054809  1.000000   0.354104  0.576223
#newspaper  0.056648  0.354104   1.000000  0.228299
#sales      0.782224  0.576223   0.228299  1.000000

sns.heatmap(df.corr(),annot=True)


# Korelasyonlar düşük diye EDA aşamasında bu değişkenler atılmamalı. Çünkü;
# .. bu değişkenler kullanılıp feature engineering yapılabilir ve o yeni oluşan
# .. feature lar işimize yarayabilir ya da feature lar bazen öğrenmede çok işe
# .. yarayabilir(korelasyonu düşük olmasına rağmen)
# Orion Hoca: Önce tüm değişkenleri atmadan modele sokup daha sonra feature çıkartmak daha iyi


#Train-Test Split

#verilerin hepsinin sayısal olması gerekiyor 


X=df.drop(columns="sales")
y=df["sales"]

from sklearn.model_selection import train_test_split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
df.sample(15)

#test_size:datanın ne kadarının test e ayrılacağına karar verir.
#amaç train setine olabildiğince max gözlem verebilmek

print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)
# 200 gözlemim vardı 0.80 ini(160 tanesini) train e ayırdı. 0.20 sini test e ayırdı train_test_split()

#Train features shape :  (160, 3)
#Train target shape   :  (160,)
#Test features shape  :  (40, 3)
#Test target shape    :  (40,)


X_train
X_test
y_train

#x_train y_train deki değerleri baz alarak kendini eğitir,kuralları belirler
#yani katsayıları belirler  
#eğitim bittikten sonra modele x_tes verilir,y_test verilmez,aldığı eğitime göre
#x_test deki değerlere tahminde bulunur(y_pred )
#y_pred=X_test sonucunda alınan tahminlerdir
#sonrasında y_pred ile y_test karşılaştırılır
#karşılaştırma sonucunda da r2,mae,rmse skorları alınacak 


#Model Fitting and Compare Actual and Predicted Labels

from sklearn.linear_model import LinearRegression
model=LinearRegression() #---->önce model oluşturulur 
model.fit(X_train,y_train)#---->sonra bu modelin içine X_train,y_train atılır,eğitime verilir
#amaç y trainde belirtilen gerçek değerlere en yakın tahminleri elde edebilmek
y_pred=model.predict(X_test) #eğitim sonucu y_pred değerleri bulunur.tahmin yapıyor
y_pred #modelin döndürdüğü tahminler


model.coef_ #modelin bütün featurelar için belirlediği katsayıları verir
 # b1,b2,b3 : katsayılar(Öğrenilen kısım diyebiliriz)
#array([0.04472952, 0.18919505, 0.00276111])
model.intercept_ # b0: y eksenini kestiği nokta

# y_pred = b3 * TV + b2 * radio + b3 * newspaper + b0


X_test.loc[95] # Rasgele bir değer aldık tahminde bulunmak için

#TV           163.3
#radio         31.6
#newspaper     52.9
#Name: 95, dtype: float64

sum(X_test.loc[95] * model.coef_) + model.intercept_ 
#y_pred in arka planda ne yaptığını göstermek için manuel hesaplandı

#16.408024203228628  X_test.loc[95] için tahmin sonucu 


my_dict = {"Actual": y_test, "Pred": y_pred, "Residual":y_test-y_pred}

comparing = pd.DataFrame(my_dict)
comparing.head(5)
#     Actual       Pred  Residual
#95     16.9  16.408024  0.491976
#15     22.4  20.889882  1.510118
#30     21.4  21.553843 -0.153843
#158     7.3  10.608503 -3.308503
#128    24.7  22.112373  2.587627

result_sample = comparing.head(25) # Grafik çizdireceğimiz için 25 tane gözlem alalım
result_sample

result_sample.plot(kind ="bar", figsize=(15,9))
plt.show()


#ERROR METRICS

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

R2_score = r2_score(y_test, y_pred) # y_pred ile y_test i karşılaştırıyoruz
R2_score # Sales i Tv,Radio ve newspaper yüzde 89 oranında açıklıyor.(Ares Hoca: Fena değil)
# Açıklanamayan kısım için feature engineering yapabiliriz.
#0.8994380241009121

mae=mean_absolute_error(y_test,y_pred)
mae# Hataları fazla cezalandırmıyor.

#1.4607567168117597

mse = mean_squared_error(y_test, y_pred)
mse   # Hataları fazla cezalandırıyor
#3.1740973539761015

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse # Yorumlaması daha kolay. Tercih ettiğimiz metric
#1.7815996615334502

sales_mean = df["sales"].mean()
sales_mean
#14.022500000000003

mae / sales_mean
#0.10417234564533852 #modelim ortalaa yüzde 10 civarında yanlış tahminler yapıyor
#yapılan kötü tahminleri gözz ardı eder

rmse / sales_mean
#0.12705292647769298

#Adjusted R2 score

def adj_r2(y_test, y_pred, X):
    r2 = r2_score(y_test, y_pred)
    n = X.shape[0]   # number of observations  
    p = X.shape[1]-1 # number of independent variables  
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2
adj_r2(y_test, y_pred, X)
#0.8984170903354392
#Eğer dataya yeni feature lar ekliyorsak,datanın yeni bilgiler öğrenebilmesi için
#yeni gözlemler ilave etmemeiz gerekir ki data yeni şeyler öğrenebilsin.
#Feature sayısı arttığında gözlem sayısını arttırmazsak R2 score da yalancı bir iyileşme olur ama MAE ve 
#RMSE de herhangi bir değişiklik olmaz veya daha çok kötüleşebilir.Bunun önüne geçmek için
#"Asjusted r2 score" fonksiyonu kullanıyoruz.Bu score gözle sayısı ile feature sayısını
#dengeler
# Biz feature engineering de feature sayısını arttırdıkça r2 de yalancı bir iyileşme oluyor
# Bunu engellemek adın r2_adj score u kullanıyoruz yorumlarken/değerlendirme yaparken
# Orion Hoca: Feature eklerseniz r square her zaman iyileşir(O yüzden gözlemde eklemeliyiz ki r2 bizi yanıltmasın)
# Feature sayınız ile gözlem oranınız aynı oranda artmalı ki r2 score umuz dengeli olsun
# .. r2_adj bunu engelliyor
# Interview larda karşılaşabilirsiniz diye bunu anlatıyoruz
# Interview soru: Lineer regresyonda feature ekleyerek doğruluğum arttı. Bunu nasıl teyit ederiz
# Cevap: r2_adj score a bakarız deriz


#What is the concept of punishment for RMSE error metric?

variables = {"Actual": [1000, 2000, 3000, 4000, 5000], "pred": [1100, 2200, 3200, 4200, 5300]}  # 6000
df_2 = pd.DataFrame(variables)
df_2

#   Actual  pred
#0    1000  1100
#1    2000  2200
#2    3000  3200
#3    4000  4200
#4    5000  5300

df_2["residual"] = abs(df_2.Actual - df_2.pred)
df_2
#   Actual  pred  residual
#0    1000  1100       100
#1    2000  2200       200
#2    3000  3200       200
#3    4000  4200       200
#4    5000  5300       300


#mae
df_2.residual.sum()/5 # 5: gözlem sayısı
#200 # 5300 yerine 6000 yazarsak sonuç 340.00 çıkacak


#rmse
((df_2.residual**2).sum()/5)**0.5  # 0.5 : Karekök almak yerine 0.5 ile çarpılmış
# 5300 yerine 6000 yazarsak bu 475.39 çıkacak

#209.76176963403032


def eval_metric(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    R2_score = r2_score(actual, pred)
    print("Model testing performance:")
    print("--------------------------")
    print(f"R2_score \t: {R2_score}")
    print(f"MAE \t\t: {mae}")
    print(f"MSE \t\t: {mse}")
    print(f"RMSE \t\t: {rmse}")
# Her seferinde yazmak yerine burada bunları bir fonksiyona topladık

eval_metric(y_test, y_pred) # Modelin görmediği data

#Model testing performance:
#--------------------------
#R2_score 	: 0.8994380241009121
#MAE 		: 1.4607567168117597
#MSE 		: 3.1740973539761015
#RMSE 		: 1.7815996615334495

#y_pred=model.predict(X_test)
#sadece test setinde aldığımız skorlarla yetinmiyoruz
#test setinde aldığımız skorların gerçekten genelleme yapılabilecek
#skorlar olduğunu anlamak için mutlaka eğitim yaptımız train setindende tahmin alırız.
y_train_pred = model.predict(X_train)
#train stinde skorlar genelde daha yüksek olur çünkü eğütimi yaptığı yer train 

eval_metric(y_train, y_train_pred) # Modelin gördüğü data
# test ile train score larımın(r2 özellikle) yakın olması gerekiyor. Yoksa
# .. overfitting ve underfitting durumları oluşuyor(Daha sonra anlatılacak)

#Model testing performance:
#--------------------------
#R2_score 	: 0.8957008271017817
#MAE 		: 1.1984678961500141
#MSE 		: 2.7051294230814147
#RMSE 		: 1.6447277656443375


#Is data suitable for linear regression?

residuals = y_test-y_pred "1"
plt.figure(figsize = (10,6))
sns.scatterplot(x = y_test, y = residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()

sns.kdeplot(residuals); # Residular normal dağılım sergiliyor mu diye bakıyoruz
# Ares hoca: Normal dağılıma yakın bir dağılıma benziyor

stats.probplot(residuals, dist ="norm", plot =plt); # Q-Q plot
# residularım bu doğruya yakın olup sarılırsa datam normal dağılım sergiliyor diyebiliriz

#https://stats.stackexchange.com/questions/12262/what-if-residuals-are-normally-distributed-but-y-is-not

from scipy.stats import skew
skew(residuals) # 3. method : -0.5 ile +0.5 aralığında residuların normal dağılım sergilediğini gösteriyor
# -1 ile 1 olursa normal dağılımdan uzaklaştığını söyleyebiliriz


from yellowbrick.regressor import ResidualsPlot # 4. method

# Instantiate the linear model and visualizer
model = LinearRegression()
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();        # Finalize and render the figure



#Prediction Error for LinearRegression

from yellowbrick.regressor import PredictionError
# Instantiate the linear model and visualizer
model = LinearRegression()
visualizer = PredictionError(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();    

#Retraining Model on Full Data

final_model = LinearRegression()
final_model.fit(X, y)

#Coefficients

final_model.coef_
#array([ 0.04576465,  0.18853002, -0.00103749])

final_model.intercept_
#2.9388893694594103

df.head()

#    TV	    radio	newspaper	sales
#0	230.1	37.8	69.2	    22.1
#1	44.5	39.3	45.1	    10.4
#2	17.2	45.9	69.3	    9.3
#3	151.5	41.3	58.5	    18.5
#4	180.8	10.8	58.4	    12.9

coeff_df = pd.DataFrame(final_model.coef_, index = X.columns, columns = ["Coefficient"] )

coeff_df
#           Coefficient
#TV            0.045765
#radio         0.188530
#newspaper    -0.001037

#Prediction on New Data

adv = [[150, 20, 15]] # 2D # Modeller 2 boyutlu olmalı o yüzden çift parantez kullanıldı
adv
#[[150, 20, 15]]

final_model.predict(adv)
#array([13.55862413])
final_model.coef_ # Bu coefficient lar yanıltıcı olabilir korelasyonla karşılaştırıldığında
# .. Burada sanki radio sanki 0.18 ile Tv(0.045) den daha önemli gibi görünüyor. Ama bu "scaling" yapmadığımız için oldu
#array([ 0.04576465,  0.18853002, -0.00103749])
final_model.intercept_
#2.9388893694594103
sum(final_model.coef_ * [150, 20, 15]) + final_model.intercept_
#13.558624130495994

adv_2 = [[150, 20, 15], [160, 20, 15]] # tv yi 10 birimlik arttırdık

final_model.predict(adv_2)    # tahmin 13.55 den 14.01 e gelmiş.
#array([13.55862413, 14.01627059])

14.01627059 - 13.55862413
# tv deki 10 birimlik artış. Benim label ımda 0.45 lik bir artışa neden oldu.
# Yani tv nin katsayısı oldu

# Biz varsayımları vs irdeledik burada ama biz her zaman bunlara bakmayıp
# .. modeli alıp fit edip vs ilerleyeceğiz
