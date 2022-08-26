#INCLASS 9

#NOTEBOOK

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline
#%matplotlib notebook
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#pd.options.display.float_format = '{:.3f}'.format

#Bu data setinde en iyi treshold'u seçmeyi öğreneceğiz.
#Elimizde dengesiz bir data seti var, skorları nasıl iyileştirebiliriz bunu göreceğiz.

df=pd.read_csv("diabetes.csv")
df.head()

#   Pregnancies  Glucose  BloodPressure  ...  DiabetesPedigreeFunction  Age  Outcome
#0            6      148             72  ...                     0.627   50        1
#1            1       85             66  ...                     0.351   31        0
#2            8      183             64  ...                     0.672   32        1
#3            1       89             66  ...                     0.167   21        0
#4            0      137             40  ...                     2.288   33        1

#Pregnancies : Kaç defa hamile kalındığı
#Glucose : Vücuttaki şeker oranı
#BloodPressure : Tansiyon
#SkinThickness : Deri kalınlığı
#Insulin : Vücudun ürettiği insulin oranı
#BMI : Vücut indexi
#DiabetesPedigreeFunction : Ailede şeker hastalığı olup olmaması durumuna göre skorlar
#Outcome : 1--> Şeker hastası, 0---> Şeker hastası değil

df.shape
#(768, 9)

#Exploratory Data Analysis and Visualization

df.info()
df.describe().T

df.Outcome.value_counts()  
#0    500
#1    268
#Name: Outcome, dtype: int64
# 1 sayısı az görünüyor ama skorlara bakmadan 'dengesizlik var' gibi bir yorum yapmıyoruz.

sns.countplot(df.Outcome);
sns.boxplot(df.Pregnancies);    
#df=df[df.Pregnancies<13]   Outlier değerler gerçekte de olabilir. 17 kere hamile kalan kadınlar var. 
#Gerçek dünya verileri olduğu için tutuyoruz ama sayıları az olduğu için atıladabilir.

sns.boxplot(df.SkinThickness);
df=df[df.SkinThickness<70]   
# Gerçekte 100 diye bir deri kalınlığı olmadığı için onu attık.
sns.boxplot(df.SkinThickness);

sns.boxplot(df.Insulin);

sns.boxplot(df.Glucose);
df=df[df.Glucose>0]    # Glukoz 0 olamaz o yüzden attık.
sns.boxplot(df.Glucose);

sns.boxplot(df.BloodPressure);
df=df[df.BloodPressure>35]    # Kan basıncı 30'un altında olamaz, o yüzden 35 altını attık.
sns.boxplot(df.BloodPressure);

sns.boxplot(df.BMI);
df=df[df.BMI>0]    # Vücut kitle indexi 0 olamaz, o yüzden attık.
sns.boxplot(df.BMI);

df.shape
#(720, 9)

df.Outcome.value_counts()     
#0    473
#1    247
#Name: Outcome, dtype: int64
# Bazı verileri attıktan sonra veri sayımız biraz düştü.

index = 0
plt.figure(figsize=(20,20))
for feature in df.columns:
    if feature != "Outcome":
        index += 1
        plt.subplot(3,3,index)
        sns.boxplot(x='Outcome',y=feature,data=df)

plt.figure(figsize=(10,8))# Multicollineraity olsa bile Ridge ve Lasso arka planda bu sorunu giderecek (Default--> Ridge)
sns.heatmap(df.corr(), annot=True);    

# df.corr()                                                         
# 1 sınıfıyla olan corr'ların görseli.
# df.corr()["Outcome"].sort_values().plot.barh()                    
# En yüksek corr ilişkisi glukoz ile.
df.corr()["Outcome"].drop("Outcome").sort_values().plot.barh();     
# Glukoz, kilo(BMI), Age yüksekse şeker hastası olma ihtimali yüksek.

sns.pairplot(df, hue = "Outcome");


#Train | Test Split and Scaling
X=df.drop(["Outcome"], axis=1)
y=df["Outcome"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

#Datada dengesizlik olduğu düşünülüyorsa hem test
#hem de train seti split işleminde eşit oranlarla 
#dağılsın diye 'statify = y' denir.
#Bu şekilde 0 sınıfının da %20'sini 1 sınıfının da %20'sini 
#test için ayırır.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Modelling
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred=log_model.predict(X_test_scaled)
y_pred_proba = log_model.predict_proba(X_test_scaled)
#pred'leri karşılaştırmak için X_test ve y_test'leri birleştirip
#pred ve pred_probayı feature olarak ekledik.
#pred_proba'da 0.5'in üstündekileri 1'e altındakileri 0'a atadığını görüyoruz.

test_data = pd.concat([X_test, y_test], axis=1)
test_data["pred"] = y_pred
test_data["pred_proba"] = y_pred_proba[:,1]
test_data.sample(3)
#     Pregnancies  Glucose  BloodPressure  ...  Outcome  pred  pred_proba
#425            4      184             78  ...        1     1       0.784
#630            7      114             64  ...        1     0       0.322
#147            2      106             64  ...        0     0       0.275
#Model Performance on Classification Tasks

from sklearn.metrics import confusion_matrix, classification_report
#Hem train hem de test setini aynı anda görmek için
#aşağıdaki fonksiyonu yazdık. Amacımız, train ile
#test datalarındaki skorları kıyaslayarak bir
#overfitting veya underfitting durumu var mı 
#bunu tespit etmek.


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

#df' de 0 ve 1 sınıfları arasında unbalance bir
#durum olabileceğini gözlemlemiştik. Aşağıdaki 
#skorlara baktığımızda bunu teyit edebiliyoruz, skorlar kötü.
#Train_Set ve Test_Set skorları birbirine yakın olduğu
#için overfitting bir durumdan da bahsedemeyiz.

eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)
#Test_Set
#[[85 10]
# [20 29]]
#              precision    recall  f1-score   support
#
#           0       0.81      0.89      0.85        95
#           1       0.74      0.59      0.66        49
#
#    accuracy                           0.79       144
#   macro avg       0.78      0.74      0.75       144
#weighted avg       0.79      0.79      0.79       144


#Train_Set
#[[337  41]
# [ 89 109]]
#              precision    recall  f1-score   support

#           0       0.79      0.89      0.84       378
#           1       0.73      0.55      0.63       198

#    accuracy                           0.77       576
#   macro avg       0.76      0.72      0.73       576
#weighted avg       0.77      0.77      0.77       576

#0 sınıfına ait skorların daha iyi, 1 sınıfına ait skorların
#daha kötü olduğunu gözlemliyoruz. Peki bunun sebebi ne?
#Çünkü 0 sınıfına ait gözlem sayısı daha fazla. 
#Bu yüzden eğitimini daha iyi yapmış.
#Cross Validate ile yukarıda aldığımız skorları teyit edeceğiz :

#    

from sklearn.model_selection import cross_validate
model = LogisticRegression()
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)

df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores                                                  
# Binary modellerdeki skorlar her zaman 1 class'ına ait skorlardır.
#    fit_time  score_time  test_precision  test_recall  test_f1  test_accuracy
#1      0.009       0.005           0.600        0.450    0.514          0.707
#2      0.011       0.008           0.643        0.450    0.529          0.724
#3      0.007       0.005           0.923        0.600    0.727          0.845
#4      0.006       0.004           0.857        0.600    0.706          0.828
#5      0.005       0.004           0.706        0.600    0.649          0.776
#6      0.005       0.003           0.647        0.550    0.595          0.741
#7      0.006       0.004           0.714        0.526    0.606          0.772
#8      0.006       0.004           0.647        0.579    0.611          0.754
#9      0.007       0.007           0.733        0.550    0.629          0.772
#10     0.005       0.005           0.625        0.500    0.556          0.719

df_scores.mean()[2:]     
# Sadece skorları görebilmek için 2. indexten sonrasına bakıyoruz.   (Scale edilmiş skorlar)
#test_precision   0.710
#test_recall      0.541
#test_f1          0.612
#test_accuracy    0.764
#dtype: float64

#Aşağıya eval_metric' i tekrar yazdıralım, scale edilmeden
#önceki skorlar(yukarıdaki) ile sonraki skorları (aşağıdaki) kıyaslayalım.
#Scale edildikten sonra skorların biraz düştüğünü gördük. 
#Ama çok bariz bir fark yok diyebiliriz. Cross Validate işlemi bu durumu tespit etmek adına önemli.
eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)
# (Scale edilmemiş skorlar)
#Yukarıdaki skorlarımızda Recall çok düşük. Amacımız, 
#Recall'ı artırmak ama Precision ile dengeli bir şekilde. 
#Dolayısıyla f1-score' da artmış olacak. Modelimiz iyileşecek.
#!!!!!!! Çok dengesiz datasetlerinde Test_Set ile Train_Set 
#arasında çok fazla fark olduğunda overfitting durumu var diyemeyiz. 
#Bu durumda bakacağımız skorlar 'macro' ve 'weighted' olmalı. 
#Overfitting olup olmadığına bu iki skorla karar verebiliriz. !!!!!!!
#Modelimizde overfitting olsa bile logisticRegression() içine penalty = l1 gibi değerler yazarak overfitting ile mücadele edebiliriz. (Default değeri l2.)

#Cross Validate for 0 class
#Eğer default class olan 1 değil de 0 sınıfı için cross validate 
#yapmak istersek 'make_scorer' fonksiyonunu import ediyoruz

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

import sklearn
sklearn.metrics.SCORERS.keys()
#Yukarıdaki skorları direk make_scorer içinde kullanamıyorum
#çünkü kullandığım zaman 1 sınıfının skorlarını hesaplıyor. 
#Biz ise 0 sınıfının skorlarını istiyoruz.

#make_scorer : 'pos_label' aslında make_scorer' ın 
#içinde geçmiyor ama make_scorer içinde kullanılan 
#fonksiyonun da içinde geçenleri kullanabilirsin diyor, 
#böyle bir esneklik sağlıyor. Mesela f1_score()'un içinde 
#geçen pos_label' ı burada kullanabileceğiz. Bu yüzden 
#make_scorer'ın içine pos_label' ı ekledik.

f1_0 = make_scorer(f1_score, pos_label =0)
precision_0 = make_scorer(precision_score, pos_label =0)
recall_0 = make_scorer(recall_score, pos_label =0)

model = LogisticRegression()
#cross_validate'de scoring'lere mse, rmse gibi 
#skorları yazıyorduk. Burda 0 sınıfına ait skorları istediğimiz için, yukarıda 
#tanımladığımız f1, precision ve recall değerlerini dict olarak 
#cross_validate'in içine veriyoruz. Böylece 0 sınıfına ait skorları alabileceğiz.

scores = cross_validate(model, X_train_scaled, y_train,scoring = {"precision_0":precision_0, "recall_0":recall_0, "f1_0":f1_0}, cv = 10)
#Bulduğumuz skorları DataFrame yapısına çevirdik :
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
#    fit_time  score_time  test_precision_0  test_recall_0  test_f1_0
#1      0.012       0.003             0.744          0.842      0.790
#2      0.011       0.007             0.750          0.868      0.805
#3      0.012       0.006             0.822          0.974      0.892
#4      0.008       0.007             0.818          0.947      0.878
#5      0.007       0.004             0.805          0.868      0.835
#6      0.008       0.005             0.780          0.842      0.810
#7      0.008       0.004             0.791          0.895      0.840
#8      0.010       0.005             0.800          0.842      0.821
#9      0.011       0.007             0.786          0.892      0.835
#10     0.007       0.005             0.756          0.838      0.795

df_scores.mean()[2:]         # Yukarıda bulduğumuz skorların ortalamasını aldık.
#test_precision_0   0.785
#test_recall_0      0.881
#test_f1_0          0.830
#dtype: float64
#Aşağıda Cross_Validate yapılmamış değerleri tekrar yazdırdık ki bir kıyas yapalım, 
#yukarıda yeni bulduğum değerlerle cross işlemi öncesi skorlarım değişmiş mi?
#Kıyaslama için bu sefer 0 değerlerine odaklanacağız. Çünkü 0 değerlerinin 
#scorlarını bulduk. (cross_validate işleminden sonra skorlarımın biraz düştüğünü gözlemliyorum.)
eval_metric(log_model, X_train_scaled, y_train, X_test_scaled, y_test)
#Test_Set
#[[85 10]
# [20 29]]
#              precision    recall  f1-score   support

#           0       0.81      0.89      0.85        95
#           1       0.74      0.59      0.66        49

#    accuracy                           0.79       144
#   macro avg       0.78      0.74      0.75       144
#weighted avg       0.79      0.79      0.79       144
#Train_Set
#[[337  41]
# [ 89 109]]
#              precision    recall  f1-score   support

#           0       0.79      0.89      0.84       378
#           1       0.73      0.55      0.63       198

#    accuracy                           0.77       576
#   macro avg       0.76      0.72      0.73       576
#weighted avg       0.77      0.77      0.77       576

#GridSearchCV
#Skorlarımız çok iyi çıkmadı. Peki bunları nasıl iyileştireceğiz?
import sklearn
sklearn.metrics.SCORERS.keys()
from sklearn.model_selection import GridSearchCV

model = LogisticRegression()
#Logistic Regression İçindeki Parametreler :
#ogisticRegression overfitting ile mücadele etmek amacıyla içine penalty = l1, l2, elasticnet parametrelerini alıyordu.
#l2 ----> Ridge
#l1 ----> Lasso
#Linear Regression'daki alpha yerine burda C parametresi var. Bu parametre alpha 
#ile ters orantılı çalışır. Alpha büyüdükçe regularization artar; C küçüldükçe 
#regularization artar (bias ekler). Yani C değerinin küçülmesi iyi bir şey.
#class_weight : Class sayıları arasında dengesizlik varsa; sayısı az olan 
#sınıfı daha çok ağırlıklandırır. Yani zayıf olan sınıfa daha çok tahmin yaptırır.
#solver : Modeller metricleri minimize etmek için 'Gradient Descent tabanlı' 
#çalışırlar. Solver metrikleri de Gradient Descent methodlarıdır. Çok bilinmiyorsa d
#efault değerlerinin değiştirilmesi önerilmez. Çoğunlukla default değeri 
#iyi sonuç verir. (solver : 'lbfgs')
#Eğer data küçükse ''solver : liblinear'', çok büyük datalarda ise ''solver : sag'' 
#veya ''solver : saga'' iyi bir seçim olabilir. Kafamızda soru işareti oluştuğu 
#zaman bunları deneyerek sonuçları karşılaştırabiliriz.
#multi_class : 0, 1, 2 diye üç sınıf olsun. ROC/AUC çizerken 2 sadece binary 
#olanları çizebiliyor. Burdaki 3 sınıfı çizmek için herhangi bir sınıfı alıp 
#geri kalanına tek bir sınıf gibi davranır. Böylece 2 sınıf varmış gibi olur. 
#Tüm ihtimaller için bunu yapar ve çizgilerini çizer. 
#multi_class = 'ovr' bunu sağlar. default = 'auto'
#Biz aşağıda bir fonksiyon tanımlayarak Ridge ve Lasso'dan hangisinin daha iyi sonuç verdiğine bakacağız :

penalty = ["l1", "l2"]                
# l1 ve l2 skorlarına bakacağız.
C = np.logspace(-1, 5, 20)            
# C parametresi logspace aralığında daha iyi sonuçlar verir. (Hangi sayının logunu aldığımda bu aralıktan bir sayı döndürür?)
class_weight= ["balanced", None]      
# Classlar arası dengeleme yapsın veya yapmasın.

# The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies 
# in the input data

solver = ["lbfgs", "liblinear", "sag", "saga"]   
# Gradient descent methodlarından hangisini kullanayım?
param_grid = {"penalty" : penalty,
              "C" : C,
              "class_weight":class_weight,
              "solver":solver}


grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=10,
                          scoring = "recall",     
                          n_jobs = -1) 
# 1 sınıfına ait en iyi recall'ı hangi parametreler getirecek? Bunu hesaplar.             
# Recall dedik çünkü skorlarımızda bu değer kötü. f1 de diyebilirdik. Sırayla denenebilir.
grid_model.fit(X_train_scaled,y_train)     # Eğitimimizi yaptık.

grid_model.best_params_     
#{'C': 0.1, 'class_weight': 'balanced', 'penalty': 'l1', 'solver': 'liblinear'}   
# Yukarıda tanımlanan modele göre çıkan en iyi parametre değerleri.
eval_metric(grid_model, X_train_scaled, y_train, X_test_scaled, y_test)
#Test_Set
#[[76 19]
# [13 36]]
#              precision    recall  f1-score   support
#
#           0       0.85      0.80      0.83        95
#           1       0.65      0.73      0.69        49
#
#    accuracy                           0.78       144
#   macro avg       0.75      0.77      0.76       144
#weighted avg       0.79      0.78      0.78       144


#Train_Set
#[[288  90]
#[ 49 149]]
#              precision    recall  f1-score   support

#           0       0.85      0.76      0.81       378
#           1       0.62      0.75      0.68       198

#    accuracy                           0.76       576
#   macro avg       0.74      0.76      0.74       576
#weighted avg       0.78      0.76      0.76       576


#Yeni sonuç ile eski sonucu kıyasladığımızda; precision değerinin düştüğünü ama 
#recall değerinin de yükseldiğini görüyoruz. f1 score da 66'dan 70'e çıkarak 
#dengeyi korumuş. Amacımıza ulaştık; recall değerini dengeli bir şekilde artırdık.

#0 skorları ise düştü. Bizim amacımız hasta olanları yani 1'leri tespit etmek.
#Bu yüzden 1 olanları iyileştirmeye yönelik parametreler kullandık.

#Tek bir modelde hem 1 hem 0' lar için skorlara bakılmaz, bu hatalı olur. 
#0 skorlarına bakıyorsak ayrı model, 1 skorlarına bakıyorsak ayrı model kullanmalıyız.



#ROC (Receiver Operating Curve) and AUC (Area Under Curve)

#ROC/AUC; birçok treshold değeri belirler ve buna göre eksende noktalar bulur. 
#(Treshold = 0.5'e göre düşman olduğunu bildim veya bilemedim gibi.) 
#Bu noktaların altında kalan alan ne kadar büyükse, model dost ile düşmanı ayırmakta o kadar başarılı demektir.

#1 sınıfını düşman, 0 sınıfını dost gibi düşünüyoruz ve amacımız düşmanı tespit etmek.

#y ekseni, düşman olarak doğru tahmin ettiklerimiz. (True Positive Rate)

#x ekseni, düşman olarak yanlış tahmin ettiklerimiz. (False Positive Rate)


from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve, average_precision_score, precision_recall_curve

#1 noktası True-Positive'in en yüksek olduğu nokta; 0 noktası ise False- Positive'in
#en düşük olduğu nokta. Amacımız ilkini yüksek, ikinciyi düşük yapmak ki alttaki alan büyüsün.

plot_roc_curve(grid_model, X_test_scaled, y_test);                # Modelin başarısı : 0.85




#Precision-Recall-Curve

#Bizim için geçerli olan yöntem : precision_recall_curve. Çünkü datasetimiz dengesiz.
#(İlk yöntem daha iyi olmasına rağmen ikinci yöntemi seçtik.)


#Finding Best Threshold
#Amacımız yeni treshold'lar belirleyerek en büyük alanı çizebilmek ve model başarısını artırabilmek.
#default olarak treshold değeri = 0.5
#Yeni tresholdlar belirleyerek mesela 0.3; 0.3 ün altındakilere 0; üstündekilere 1 diyeceğiz.

#ROC ve AUC ile Best Treshold :

#Bunun için en iyi treshold değerini bulacağız yani alttaki alanın en büyük olduğu treshold değeri.
#!!!!!! Dengeli datasetlerinde ROC / AUC, dengesiz datasetlerinde Precision Recall Curve kullanılır. !!!!!!!!
#!!!!!!!!! Best treshold sadece train setinde bulunur. Eğer test setinde de denersek 'data leakage(kopye)' olur. !!!!!!!!

plot_roc_curve(grid_model, X_train_scaled, y_train);   
y_pred_proba = log_model.predict_proba(X_train_scaled)  
# Train setindeki predict_proba' yı aldık ki yukardaki grafikteki skorla karşılaştırabilelim.
roc_auc_score(y_train, y_pred_proba[:,1])      
#0.8378493934049489     
# roc_ouc_score içine eğittiğimiz y yi ve y_train'den aldığımız proba'nın 1 sınıfı için olan değer

#Değerimiz yukarıdaki grafikte 0.84 çıkmıştı predict işleminde de 0.83 çıktı. Birbirine yakın değerler elde ettik.
#fp_rate : False - Positive Rate (Amaç minimum yapmak). (FPR)
#tp_rate : True - Positive Rate (Amaç maximum yapmak). (TPR)
#treshold : 0 - 1 arasında aldığı olasılıklar.

#fp_rate    # Her bir treshold'a göre aldığı olasılık değerleri.

#tp_rate     #Her bir treshold'a göre aldığı olasılık değerleri.

fp_rate, tp_rate, thresholds = roc_curve(y_train, y_pred_proba[:,1])


#(max TPR) - (min FPR) çıkarırsak; burası düşmanın en iyi tespit edildiği noktadır.
#(Düşmana düşman dediğim max değerden, dosta düşman dediğim min değeri çıkardım.)

optimal_idx = np.argmax(tp_rate - fp_rate)          
# İçerideki max değer neyse onun index nosunu döndürür.
optimal_threshold = thresholds[optimal_idx]         
# Bulunan indexi tresholdun içine verdik. En optimal treshold'u bize döndürür.
optimal_threshold
#0.33938184887578754

#!!!!! Best treshold için ROC ve AUC da kullanabiliriz,
#Precision-Recall-Curve da kullanabiliriz. Aynı sonuçlar çıkar, 
#sadece hesaplamaları farklı. (ROC AUC mantığı daha kolay) !!!!!

#Precision-Recall-Curve ile Best Treshold :

plot_precision_recall_curve(grid_model, X_train_scaled, y_train);   

y_pred_proba = log_model.predict_proba(X_train_scaled)
average_precision_score(y_train, y_pred_proba[:,1])
#0.7120696300524079

precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred_proba[:,1])

optimal_idx = np.argmax((2 * precisions * recalls) / (precisions + recalls))
optimal_threshold = thresholds[optimal_idx]
optimal_threshold
#0.33938184887578754


grid_model.predict_proba(X_test_scaled)[:,1]    
# 0.5 treshold'a göre dönen değerler.

#Biz aşağıdaki fonksiyonda artık 0.5 değil de yeni bulduğumuz best treshold olan 0.33'e göre değerler döndüreceğiz.
#Önce seri içine yazdık yoksa apply fonk. uygulayamazdık

#Aldığımız değer yeni treshold'dan büyükse (0.33), 1 sonucunu döndür; değilse 0 döndür.

y_pred2 = pd.Series(grid_model.predict_proba(X_test_scaled)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)

print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))
#[[55 40]
# [ 3 46]]
#              precision    recall  f1-score   support

#           0       0.95      0.58      0.72        95
#           1       0.53      0.94      0.68        49

#    accuracy                           0.70       144
#   macro avg       0.74      0.76      0.70       144
#weighted avg       0.81      0.70      0.71       144

#Yukarıdaki sonuçlara baktığımızda 1 class'ına ait precision değerleri hemen 
#hemen aynı ama recall değeri baya yükseldi.

#Aşağıda yeni treshold değeri ile train setine de baktık. 
#Orda da 1 sınıfına ait recall değerlerinin iyileştiğini görüyoruz.

y_train_pred2 = pd.Series(grid_model.predict_proba(X_train_scaled)[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
print(confusion_matrix(y_train, y_train_pred2))
print(classification_report(y_train, y_train_pred2))

#[[196 182]
# [ 16 182]]
#              precision    recall  f1-score   support

#           0       0.92      0.52      0.66       378
#           1       0.50      0.92      0.65       198

#    accuracy                           0.66       576
#   macro avg       0.71      0.72      0.66       576
#weighted avg       0.78      0.66      0.66       576

#Aşağıdaki fonksiyon, treshold'u ile oynanmış bir dataya Cross Validation'ın 
#arkada yaptığı işlemin manual olarak yapılması. LogisticRegression'da yapılan işlemleri içeriyor :


from sklearn.model_selection import StratifiedKFold    # Modeli kaç parçaya ayırmak istiyorsak ona göre index numaraları belirler.

def CV(n, est, X, y, optimal_threshold):
    skf = StratifiedKFold(n_splits = n, shuffle = True, random_state = 42)
    acc_scores = []
    pre_scores = []
    rec_scores = []
    f1_scores  = []
    
    X = X.reset_index(drop=True)       # Index no'ları her işlemden sonra sıfırlaması için.
    y = y.reset_index(drop=True)
    
    for train_index, test_index in skf.split(X, y):
        
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
        
        
        est = est
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        y_pred_proba = est.predict_proba(X_test)
             
        y_pred2 = pd.Series(y_pred_proba[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0)
        
        acc_scores.append(accuracy_score(y_test, y_pred2))
        pre_scores.append(precision_score(y_test, y_pred2, pos_label=1))
        rec_scores.append(recall_score(y_test, y_pred2, pos_label=1))
        f1_scores.append(f1_score(y_test, y_pred2, pos_label=1))
    
    print(f'Accuracy {np.mean(acc_scores)*100:>10,.2f}%  std {np.std(acc_scores)*100:.2f}%')
    print(f'Precision-1 {np.mean(pre_scores)*100:>7,.2f}%  std {np.std(pre_scores)*100:.2f}%')
    print(f'Recall-1 {np.mean(rec_scores)*100:>10,.2f}%  std {np.std(rec_scores)*100:.2f}%')
    print(f'F1_score-1 {np.mean(f1_scores)*100:>8,.2f}%  std {np.std(f1_scores)*100:.2f}%')


model = LogisticRegression(C= 0.1, class_weight= 'balanced',penalty= 'l1',solver= 'liblinear')  
CV(10, model, pd.DataFrame(X_train_scaled), y_train, optimal_threshold)   
#Accuracy      64.76%  std 5.47%
#Precision-1   49.86%  std 4.68%
#Recall-1      91.45%  std 5.95%
#F1_score-1    64.25%  std 3.31%
# Bulduğumuz C değeri ve kullandığımız parametreler neyse yazmalıyız.
# Scale edilmiş data array' e dönüştüğü için burda tekrar DataFrame'e dönüştürüyoruz.

#n_split : Data setini 10'a böl 9' unu train, 1' ini test seti yapar. 9 tane train'in indexlerini belirler.
#test_index : Test seti için ayırdıklarının index no'su.
#Bu index no'lara göre yukarıdaki fonksiyondaki for döngüsüne girer. Index no' lara göre 
#X_train, y_train, X_test, y_test değerlerini belirler. Her for döngüsünde train ve test setleri değişir.
#est = est kısmında modeli eğitir ve pred ve predict_proba' ları alır.
#y_pred2 = pd.Series(y_pred_proba[:,1]).apply(lambda x : 1 if x >= optimal_threshold else 0) 
#kısmında ise daha önce yukarda yaptığımız gibi optimal treshold değerlerini bulur.
#Bulduğu her değeri acc_scores = [], pre_scores = [], rec_scores = [], f1_scores = [] içine atar.
#print kısmında ise bulunan değerlerin ortalamasını alır.

#Final Model and Model Deployment

scaler = StandardScaler().fit(X)
    
import pickle
pickle.dump(scaler, open("scaler_diabates", 'wb'))   
    
X_scaled = scaler.transform(X)

final_model = LogisticRegression(class_weight = "balanced").fit(X_scaled, y)

pickle.dump(final_model, open("final_model_diabates", 'wb'))

X.describe().T


my_dict = {"Pregnancies": [3, 6, 5],
           "Glucose": [117, 140, 120],
           "BloodPressure": [72, 80, 75],
           "SkinThickness": [23, 33, 25],
           "Insulin": [48, 132, 55],
           "BMI": [32, 36.5, 34],
           "DiabetesPedigreeFunction": [0.38, 0.63, 0.45],
           "Age": [29, 40, 33]
          }

sample = pd.DataFrame(my_dict)
sample

scaler_diabates = pickle.load(open("scaler_diabates", "rb"))

sample_scaled = scaler_diabates.transform(sample)
sample_scaled

final_model = pickle.load(open("final_model_diabates", "rb"))

predictions = final_model.predict(sample_scaled)
predictions_proba = final_model.predict_proba(sample_scaled)
predictions2 = [1 if i >= optimal_threshold else 0 for i in predictions_proba[:,1]]


sample["pred_proba"] = predictions_proba[:,1]
sample["pred_0.50"] = predictions
sample["pred_0.34"] = predictions2
sample

