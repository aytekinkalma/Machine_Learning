#LESSON 10 

#K Nearest Neighbors Theory (KNN)

#*'A lazy lenear' bir algoritmadir 
#*Training data yoktur 
#*Butun datayi hafizasina alir,yeni gelen data nereye gelirse en yakinindaki 
#komsularina gore o datayi siniflandirir 
#Linear regression ve logistic regression parametric bir algoritmalardi.Fakat,KNN 
#non-parametric bir algoritmadir.Yani b1 ve bo gibi katsayilar yoktur.
#Hicbir hesaplama yapmaz

#KNN genelde az verili data setlerinde kullanilir.Buyuk data setleri icin uygun 
#degildir.
#Low dimensional datasets,Fault detection,Recommender systems gibi alanlarda
#kullanilir


#Hyper Parameters (k=5,weights='uniform',metric='minkowski',p)

#1- "k"=>Sample in khangi class a atilacagina k komsuluk sayisina gore karar 
#verir.Bu yuzden k secimi cok onemlidir.Ornegin,k=5 verilirse en yakin 5 
#komsuya bakar,encok hangi class tan eleman var ise sample i o class a atar 
#*eger k degeri cok buyuk secilirse "large bias" durumu ortaya cikar.Underfit 
#durumu ortaya cikar 

#Eger k degeri cok kucuk secilirse overfit durumu ortaya cikar.Train set te cok 
#iyi bir basari elde eder ama test datasinda cok kotu tahminler yapar 
 
#Bu yuzden optimal k degeri bulmak onemlidir.Optimal k yi bulabilmek icin,
#train-test datalari bolunur.K icin bir aralik verilir ve bu her k degeri icin 
#Cross Validate islemi yapilarak en az hata veren k degeri secilir 


#**ELBOW METHOD**
#K icin verilen araliktaki her k degeri icin error metrikleri hesaplanir.Bu 
#error metriklerine gore bir grafik elde edilir 
#KNN metodunda Grid Search islemi yerine elbow metodu ile cizilen grafige 
#gore karar vermek daha mantiklidir 

#Gridhsearch islemi k yi 20-25 degerleri arasina goturur 

#2 Weights =>{'uniform','distance'}
#uniform(default deger)=>her bir komsunun 1 oyu var 

#distance=>Yakin olan komsunun biraz daha uzak olana ustunlugu olur.Yakin olanin 
#oyu artar.Bu sekilde tek bir komsu,birden fazla komsudan daha agir oldugu icin 
#ustunluk kazanabilir 

#3 Metric=>Mesafe hesabi yapan yaparetme.(hem kategoric hem cotinius olan)
#datalarda kullanilir

#Euclidean Distance
#Manhattan Distance   =>bu ikisinin karisimi olan ("minkowski") default parametre

#a
#|            eucllidean distance=|ac| uzunlugu
#|            manhattan distance=|ab|+|bc| uzunlugu
#|________
#b        c

#4-  p=>Minkowski parametresi icin kullanilan bir parametre

#p=1 =>Manhattan distance gibi davranir 
#hatalari cezalandirdigi icin outlier lara karsi direncli 
#daha cok multimensional data setlerinde kullanilir
#(feature sayisi>5 olan data setlerinde)

#p=2 =>Euclidean distance gibi davranir 
#outlier lar ile mucadelelerde iyi degil.Daha cok kucuk data setlerinde kullanilir
#(feature sayisi<5 olan data setlerinde)

#!!!!!!KNN de scale islemi zorunlu.Cunku mesafe tabanli bir algoritma

#Olumlu yonleri>
#*Lazy learner oldugu icin assumptionlar ile ilgilenmez.Bu yuzden her datada 
#kullanilabilir
#Anlamasi ve uygulamasi kolaydir
#Training yoktur
#Distance metricleri degistirilirse farkli sonuclar elde edilebilir
#Hem classification hem de regression analizlerinde kullanilabilir

#Olumsuz yonleri:
#Buyuk datalarda kullanissiz(Tum noktalar tek tek hesaplandigi icin uzun surer
#Outlierlardan cok etkilenir(Dengesiz data setleri icin uygun degil)
#Overfit tehlikesinden dolayi k secimi cok onemli
#Scalling islemi zorunludur

#regression analizlerinde de yine komsularina bakar.Bunlarin ortalamasina bakarak
#sample i bir class a atar 
#classification=>Komsularin moduna bakar 
#regression=>komsularin ortalamasina bakar


#%% ML-10-Session
# ??nceki dersin ??zeti
# Accuracy: Balanced datasetlerinde kullan??yor. Unbalanced da hangi s??n??f ??o??unluktayda onu daha ??ok yakal??yor
# Recall, Precision, F1: Unbalanced verisetlerinde bu metriklere bak??l??r.
# ROC/AUC: Her bir threshold dikkate al??narak ??iziyor. Modelimiz s??n??fland??rmay?? ne derece d??zg??n yap??yor. Bunun hakk??nda bilgi veriyordu

# KNN Theory
# Veri etraf??ndaki noktalara bakarak hangi s??n??fa ait oldu??unu bulma mant??????yla ??al??????r.
# Distance-based modeldir. Bundan dolay?? scaling yapmak gereklidir.

# KNN genel ??zellikleri
# Classification algoritmas??. Regression da yap??yor ancak biz classification modelini g??rece??iz.
# Regression modeli tercih edilmiyor
# E??itime ihtiya?? duymayan lazy learner bir modeldir. Her bir data noktas??n?? haf??zas??na al??r buna g??re map leme yapar
# .. Yeni bir data noktas?? geldi??inde o data noktas??n??n di??erlerine olan uzakl??????n?? hesapl??yor
# Non-lineer ve non-parametrictir. katsay??s?? yoktur.Bir varsay??m?? yoktur

# 1-0 ??eklinde s??n??fland??rmay?? nas??l yap??yor?
# ??rne??in y??ld??z g??zlemini d??????nelim, mavilere mi atacak, 
# turunculara m?? atayacak bu noktay??
# k : kom??u say??s??
# k se??imi ??nemli. En yak??n ka?? kom??uya bakaca????m??z?? belirliyoruz k ile.
# k n??n se??iminin d??????nda burada mesafe de ??nemli.

# Alttaki problemi nas??l s??n??fland??raca????z bakal??m bir sonraki slight a

# Slightlarda 3 farkl?? nokta i??in kom??uluklar??na bakm??????z.
# Y??ld??z i??in
    # k=1 ise en yak??n??ndaki de??er turuncu renk oldu??u i??in turuncu s??n??f??na atayacak 
    # k=3 oldu??unda en yak??ndaki 3 kom??uda(noktada) a????rl??kl?? olan s??n??f mavi oldu??u i??in maviye atad??
    # k=5 oldu??unda en yak??ndaki 5 kom??uda(noktada) a????rl??kl?? olan s??n??f mavi oldu??u i??in maviye atad??
# ??arp?? i??in
    # k=1 ise en yak??n??ndaki de??er turuncu renk oldu??u i??in turuncu s??n??f??na atayacak 
    # k=3 oldu??unda en yak??ndaki 3 kom??uda(noktada) a????rl??kl?? olan s??n??f mavi oldu??u i??in maviye atad??
    # k=5 oldu??unda en yak??ndaki 5 kom??uda(noktada) a????rl??kl?? olan s??n??f mavi oldu??u i??in maviye atad??
# Ye??il top i??in
    # Benzer mant??kta
    
# ??zetle: k se??imine g??re s??n??fland??rma sonu??lar??m??z de??i??ebiliyor. k se??imi ??nemli

# k se??imi ??nemli dedik
# k y?? b??y??k de??er se??ersek underfit (high bias). k artt??k??a tranin ve validation hatalar?? art??yor grafikte g??rd??????m??z gibi
# k y?? k??????k de??er se??ersek overfit oluyor (low bias)
# Ares Hoca: Alttaki grafi??i akl??m??zda tutal??m bu i??in mant??????n?? anlat??yor
# knn k??????k datasetlerinde kullan??l??r. ????nk?? b??y??k k de??eri se??ti??imizde b??y??k hesaplama maliyetlerine katlanmak zorunday??z

# K se??imi i??in 2 metod var
    # 1.elbow metodu
    # 2.grid search
# Error rate e bakaca????z. Bu skor accuracy, precision, recall olabilir
# 1-accuracy de??erini hesaplayaca????z. Bu bizim error rate i olacak. Yani accuracy:0.92 ise hata rate imiz: 0.08 olacak
# Miminum hata oran??n?? bulmam??z gerekiyor
# Mesela burada k =35,37,38 de vs hata d??????k g??r??n??yor ama k y?? b??y??k se??eriz bu da underfit e gitmeye sebep olur
# Ancak ??rne??in k=18 e bakt??????m??zda hatadaki de??i??im ??ok az oluyor. O y??zden k=18 de hata biraz daha y??ksek olmas??na ra??men
# .. k=18 i k=36 ya tercih etmek daha mant??kl??d??r. ????nk?? k=36 hesaplama maliyetimiz var
# Buradaki Bias-variance dengesini u sa??lamal??y??z
# Not olarak k=12 vs de se??ilebilirdi. Bu de??er de al??nabilir. Farkl?? de??erlerde al??nabilir. Bunlar?? deneyip karar verece??iz

# Elbow da   : K??r??l??m(Dirsek) yerine bakarak yorumlayaca????z. Uygulamada g??rece??iz
# Grid search: Genelde hatan??n minimum oldu??u noktay?? bulur. Se??imimiz bizim k=18 olacakken, grid search k=34 se??er
# .. kaynaklarda grid search ??n buldu??u k de??erinin kullan??lmamas?? gerekti??ini g??r??r??z. Grafikte detayl?? g??rece??iz
# Genelde elbow metodu ile karar verece??iz

# k haricinde bir di??er(ikinci) hiperparametremiz weight
# weight: modelin g??zlemleri nas??l a????rl??kland??raca????
    # uniform: B??t??n g??zlemleri e??it sekilde a????rl??kland??r??r
        # k =5 oldu??unda t??m de??erler e??it a????rl??kl?? olursa hangi s??n??f fazla olursa o s??n??fa atama yapacak.
    # distance : Mesafeye g??re a????rl??kland??rma. Yak??n g??zleme daha b??y??k bir a????rl??k veriyor
        # k=3 oldu??unda bir de??er yak??n olsun noktam??za(0 a ait), iki de??er uzak olsun(1 e ait)
        # .. 0 a ait olan??n a????rl?????? 1.8, 1 lere ait olan 0.7 ve 0.6 olursa bu noktay?? 0 s??n??f??na atayacakt??r
    # NOT: Modelden hangisinin daha iyi oldu??u de??i??ecektir

# 3. hyperparametri??imiz de Euclidian ya da manhattan
# Mesafeleri nas??l ??l??ece??iz
# Default olarak model Minkowski yi kullan??r
# p = 2 se??ersek mesafe metri??imiz euclidian .  Dik ????gendeki(3-4-5 olsun) hipoten??s?? hesaplayacak gibi d??????nebiliriz (Yani 5)
# p = 1 se??ersek mesafe metri??imiz Manhattan olacak . Dik ????gendeki kenarlar?? topluyor gibi d??????nebiliriz(Yani 3+4=7)

# Manhattan outlier lara kar???? iyi m??cadele eder mesafeyi daha fazla hesaplad?????? i??in
# Euclidian da outlier lara kar???? o kadar hassas de??il ku?? bak?????? mesafeye bakt?????? i??in

# 1. ders sonlar?? 2. ders ba??lar??
# Distance-based model oldu??u i??in scale yapmam??z gerekiyor demi??tik
# Bunu yapmazsak yanl???? tahminlerle kar????la????r??z
# Alttaki grafiklere bakal??m. X1 ve X2 feature lar?? var
# Soldaki grafikte X1 feature ??m??z  -2 ye +2 aral??????nda gibi g??r??n??yor. Range=4 birim. X1 in de??i??imi 4 birimlik alanda de??i??iyor
# .. X2 nin de -15 ile 50 aras??nda 65 birimlik bir de??i??im var. ??rne??in siyah noktam??z?? d??????nelim
# .. Acaba bu noktay?? hangi noktaya atar? k=10 olsun. X1 in range i k??????k oldu??u i??in de??erleri X1 e g??re de??erlendirecek X2 den ziyade
# .. ve buna g??re bir s??n??fland??rma yapacak modelimiz. Bu da yanl???? s??n??fland??rma yapmam??za sebep olur
# Sa??daki grafikte scale ettikten sonraki halini g??r??yoruz. Bu ??ekilde scale edilmi?? halde s??n??fland??rma yapmam??z daha do??ru

# Avantajlar??
    # Bir varsay??m?? yoktu
    # Anlamas?? ve yorumlamas?? kolay bir model
    # E??itime ihtiya?? duyulmuyor. "Map" leme yap??yor
    # Mesafeyi nas??l a????rl??kland??rd??????m??z ??nemli
    # Regresyon ve Classification problemlerinde kullan??l??r

# Dezavantajlar??
    # ??ok boyutlu veri setlerinde iyi ??al????m??yor. Feature say??s??n??n az oldu??u datasetlerinde ??al??????yor
    # Outlier lara ve dengesiz verisetlerine(??o??unluk olan s??n??fa atama ihtimali y??ksek) kar???? hassas ????nk?? distance-based model oldu??u i??in
    # k se??iminin dengesi ??nemli
    # Scale edilmesi gerekli bir model
        
# 1. Memory-based yakla????m ile s??n??fland??rma yapar
# 2. Veri b??y??k??e hesaplama maliyetimiz artar
    # Yeni datan??n B??T??N datalara olan uzakl??????n?? hesaplay??p sonra k say??s??na g??re se??im yap??yor.
    # .. Bu y??zden hesaplama maliyeti b??y??k
# Sonu?? olarak, 2 side do??ru        
        
######## K-Nearest Neighbors(KNN)   
#Ares Hoca: KNN dedi??imiz zaman akl??m??za gelmesi gerekenler;
    # Ideal for small datasets
    # Scaling data is important
    # Distance based model        
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (10,6)
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

df = pd.read_csv('gene_expression.csv')
df.head()
#   Gene One  Gene Two  Cancer Present
#0     4.300     3.900               1
#1     2.500     6.300               0
#2     5.700     3.900               1
#3     6.100     6.200               0
#4     7.400     3.400               1
# 1 ler kanser 0 lar kanser de??il
# Kanserli olup olmad??????n?? tahmin edece??iz.

#Exploratory Data Analysis and Visualization

df.info() # Missing value yok, dtype lar n??merik

df.describe() # std>mean gibi bir durum yok. Outlier ??m??z yok diyebiliriz

df["Cancer Present"].value_counts()  # Balanced bir data g??r??yoruz
#1    1500
#0    1500
#Name: Cancer Present, dtype: int64
ax= sns.countplot(df["Cancer Present"]);
ax.bar_label(ax.containers[0])

sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df,alpha=0.7)
# Grafikte datan??n baz?? noktalarda i?? i??e girdi??ini g??r??yoruz. Orada modelimiz yanl???? tahminler yapacakt??r
# Bunu bir alt grafikte daha yak??n g??zlemleyelim
#Yukaridaki grafikteki 2-6 ve 4-8 araliklarinda grift bir durum var. 
#Bu noktalara zoom yaparak baktik. Modelimizin bu noktalarda hata 
#Yapma olasiligi yuksek. KNN model, komsuluga bakarak class atamasi 
#yaptigi icin boyle grift datalarda cok iyi sonuc vermeyebilir 
# ??stteki grafikle belli bir noktaya yak??nla??t??rm???? halini inceliyoruz
sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=df, alpha=0.7, style= "Cancer Present")
plt.xlim(2,6)
plt.ylim(4,8)

sns.pairplot(data=df, hue="Cancer Present", height=4, aspect =1)
# S??n??flar??n i?? i??e girdi??ini g??r??yoruz(Sol ??st ve sa?? alttaki grafiklerde kde grafiklerinde)

sns.boxplot(x= 'Cancer Present', y = 'Gene One', data=df)
# 0 ve 1 class ??nda outlier g??r??nm??yor
# Ares hoca: Yorum olarak Gene one b??y??kse kanser olma durumu art??yor diyebiliriz

sns.boxplot(x= 'Cancer Present', y = 'Gene Two', data=df)
# Gene two data noktas?? daha b??y??kse kanser olmama durumu daha y??kse diyebiliriz

sns.heatmap(df.corr(), annot=True);
# Multicolliniearity g??r??nm??yor
# Gene one ??n say??sal de??eri artt??k??a kanser olma ihtimali art??yor(0.55)
# Gene two i??in tersi denebilir(-0.69)
# Ufak insightlar ????kard??k

from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Gene One'], df['Gene Two'], df['Cancer Present'],c=df['Cancer Present']);
# 3 feature varsa bu ??ekilde grafikler ??izdirebiliriz
# 3 boyutlu olunca data noktalar??n??n birbirinden nas??l ayr??ld??????n?? net olarak g??r??yoruz

# EDA a??amas??ndan sonra modellemeye ge??iyoruz

######### Train|Test Split and Scaling Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

######### Modelling
#PARAMETERS
#N_NEIGHBORS ---> En yakindaki kac komsuya gore atama islemi yapilsin? 
#(Default=5) Binary modellerde k sayisini tek sayi secmek makuldur.
#
#WEIGHTS ---> Komsularin uzak ve yakin olmasina gore agirliklandirma islemi yapar. 
#2 cesidi vardir :
#
#'uniform' --> Butun class' larin oyu esittir. (Default deger)
#
#'distance' --> Yakin olan komsunun uzak olana ustunlugu vardir, daha fazla 
#agirliklandirilir. Bu yuzden bir komsu, diger komsudan agir oldugu icin ustunluk
# kazanabilir.

#METRIC ---> Mesafe hesabi yapan parametredir. Euclidean Distance(kus ucusu mesafe) 
#ve Manhattan Distance' in karisimi olan minkowski parametresi default degeridir.

#P ---> Minkowski parametresi icin kullanilan bir parametredir. p=1 secilirse
# Manhattan Distance gibi davranir; p=2 secilirse Euclidean Distance gibi davranir. (Default=2)

#Manhattan Distance, hatalari cezalandirdigi icin outlier' lar ile mucadele eder.
# Feature sayisi 5' ten fazla olan datasetleri icin uygundur.

#Euclidean Distance, outlier' lar ile mucadelede iyi degildir. Feature 
#sayisi 5' ten kucuk datasetleri icin uygundur.

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5) # k=5 se??tik. Normalde deafult
#de??eri 5 zaten.
knn_model.fit(X_train_scaled,y_train)   
# Modelimizde burada train datam??z?? fit ediyoruz. E??itim derken ??t??n data noktalar??n?? map liyor asl??nda
#KNN egitim isleminde hicbir hesaplama yapmadan komusuluk sayisina gore sayisi
#en fazla olan class' a gozlemleri yerlestirir (Lazy learning). Cok buyuk 
#datalarda maliyetli bir modeldir. Cunku butun data icin gozlemler arasi 
#mesafeleri teker teker olcer. Bu yuzden daha cok kucuk datalarda tercih 
#edilen bir modeldir
y_pred = knn_model.predict(X_test_scaled)
y_pred
#predict_proba isleminde n_neighbors=5' e gore 5 sinifi sayar. Mesela 895. 
#sample' a bakarsak; bu sample' in cevresindeki 5 komsudan 2 tanesi 0 sinifina 
#ait, 3 tanesi ise 1 sinifina aitmis. Bu yuzden bu sample' i 1 sinifina atmis 
#(Bu atama islemi default deger olan 'uniform' a gore yapilmis. 
#Eger distance' a gore olsaydi komsulari agirlik degeri degisecegi icin 
#sonuclar da farklilik gosterecekti) :

y_pred_proba = knn_model.predict_proba(X_test_scaled)
# Acaba ne kadar bir olas??l??kla yapm???? tahminlerimizi bak??yoruz

pd.DataFrame(y_pred_proba)  # ??rne??in 0. indexte 0 class ??na 0.0 olas??l??kla atam????, 1 class ??na 1.0 olas??l??kla atam????

my_dict = {"Actual": y_test, "Pred":y_pred, "Proba_1":y_pred_proba[:,1], "Proba_0":y_pred_proba[:,0]}
# 1 s??n??f??na ait olma ve 0 s??n??f??na ait olma olas??l??klar?? al??yoruz bu ad??mda
# ??lk sat??ra bakarsak Ger??ek de??eri 0 pred i 0 olarak tahmin etmi?? 0.8 olas??l??kla vs... 

# Class chat soru : Hocam probay?? weighted distance  a g??re mi belirliyor? --> Ares Hoca: Evet

pd.DataFrame.from_dict(my_dict).sample(10)
#      Actual  Pred  Proba_1  Proba_0
#1569       1     1    1.000    0.000
#2631       0     0    0.000    1.000
#2596       0     0    0.000    1.000
#1288       0     0    0.000    1.000
#63         0     0    0.200    0.800
#2706       1     1    1.000    0.000
#940        1     1    1.000    0.000
#2929       1     1    0.600    0.400
#2519       0     0    0.000    1.000
#1920       1     0    0.400    0.600
############ Model Performance on Classification Tasks
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
confusion_matrix(y_test, y_pred)
plot_confusion_matrix(knn_model, X_test_scaled, y_test);
# 439 ve 396 ??m??z bizim TRUE de??erlerimizdi. 31 ve 34 False de??erlerdi
# Modelimiz 31+34 = 65 tane g??zlemi yanl???? tahmin etmi??

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#[[439  31]
# [ 34 396]]
#              precision    recall  f1-score   support
#
#           0       0.93      0.93      0.93       470
#           1       0.93      0.92      0.92       430
#
#    accuracy                           0.93       900
#   macro avg       0.93      0.93      0.93       900
#weighted avg       0.93      0.93      0.93       900
# Balanced data seti oldu??u i??in accuracy ye bakabiliriz.0.93
# Recall ve precision a bakmam??za gerek yok burada ????nk?? balanced bir data seti

y_train_pred = knn_model.predict(X_train_scaled)
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))
#y_train_pred))
#[[ 972   58]
# [  61 1009]]
#              precision    recall  f1-score   support
#
#           0       0.94      0.94      0.94      1030
#           1       0.95      0.94      0.94      1070
#
#    accuracy                           0.94      2100
#   macro avg       0.94      0.94      0.94      2100
#weighted avg       0.94      0.94      0.94      2100
# Accuracy burada y??zde 94. Skorlar birbirine yak??n. Overfit durumu g??r??nm??yor

############ Elbow Method for Choosing Reasonable K Values
#Optimal k degerini bulmak icin Elbow metodunu veya GridSearch' u kullanmamiz gerekiyor :
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
test_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k) # k y?? 1 den 30 a kadar de??i??ecek
    knn_model.fit(X_train_scaled,y_train) 
    y_pred_test = knn_model.predict(X_test_scaled)   # farkl?? k lara g??re tahminler alacak
    test_error = 1 - accuracy_score(y_test,y_pred_test) # balanced data oldu??undan error olarak accuracy ??zerinden hesaplama yap??yoruz
    test_error_rates.append(test_error)  # Bu hatalar?? yukardaki listemize ekliyoruz   
# Class chat soru: Burada 30 de??erine nas??l karar veriyoruz? --> Orion hoca: Deneme
# k y?? artt??rd??????m??zda model underfit e do??ru gidecektir(??rne??in 30 yerine 300 yazarsak deneyebiliriz(Altta denendi??inde ????kt??y?? g??rebiliriz))

plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
plt.hlines(y=0.050, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
plt.hlines(y=0.057, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
# Optimal k y?? 9 se??ebiliriz. Daha b??y??k de??erlerde daha d??????k de??erler var (22 de mesela)
# .. Ancak hatalar aras??ndaki ??ok k??????k bir de??i??im i??in 22 yi se??mek mant??kl?? olmayacakt??r

# Class chat soru: hocam bu train errorlar?? m?? sadece? --> Orion Hoca: Test
    # Thread devam??: test ile traini kar????la??t??rarak yapacak herhalde devam??nda? --> Orion Hoca: Evet
#X ekseni, 1-30 arasi k degerleri, y ekseni her k' ye denk gelen error degerleri. 
#k=9' a gelen kisim ile k=22' ye gelen kisim arasinda cok az bir error farki var. 
#Bu kadar az bir error farki icin modelin comlexity' sini 9' dan 22' ye cikarmaya 
#deger mi bunu dusunmek gerekir. Bu islem ile extra hesaplama masrafi cikar :
############# Overfiting and underfiting control for k values
# 3. ders ba????
# Train hatalar??na da bakal??m . Kodlar benzer
test_error_rates = []
train_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled,y_train) 
    y_pred_test = knn_model.predict(X_test_scaled)
    y_pred_train = knn_model.predict(X_train_scaled)
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    train_error = 1 - accuracy_score(y_train,y_pred_train)
    test_error_rates.append(test_error)
    train_error_rates.append(train_error)

plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.plot(range(1,30), train_error_rates, color='green', linestyle='--', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
plt.hlines(y=0.050, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
plt.hlines(y=0.057, xmin = 0, xmax = 30, colors= 'r', linestyles="--")
# Mavi noktalar train hatalar??
# K??rm??z?? noktalar test hatalar??
# Bunlar?? birbirine yakla??t?????? nokta 9 noktas?? gibi g??r??n??yor. Burada da farkl?? bir bak???? a????s??yla insight elde ettik

########### Scores by Various K Values
def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("Test_Set\n")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set\n")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

knn = KNeighborsClassifier(n_neighbors=1)  # ??imdi 1 kom??ulu??unda deneyelim train ve test skorlar??ma bakal??m
knn.fit(X_train_scaled,y_train)
print('WITH K=1\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# Train de accuracy 0.98  gelmi?? , test te 0.89 olmu?? accuracy . yani overfit olmu?? k k??????k iken

knn = KNeighborsClassifier(n_neighbors=22)  # k =22 de hatam d??????kt?? ama bakal??m test ve train e
knn.fit(X_train_scaled,y_train)
print('WITH K=22\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# train de accuracy:0.93, test te 0.95. De??erler yak??n. Tercih edilebilir ama bir de 9 de??erine bakal??m

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train_scaled,y_train)
print('WITH K=9\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# Train 0.94 e test 0.94 ????kt??. De??erler birbirine ??ok yak??n ????kt?? ve bunu k=9 yani d??????k bir k de??eriyle elde ettik
# 25+29= 54 tane hatal?? tahmin yapm??????z

knn = KNeighborsClassifier(n_neighbors=15)  # Bunu da deneyelim
knn.fit(X_train_scaled,y_train)
print('WITH K=15\n')
eval_metric(knn, X_train_scaled, y_train, X_test_scaled, y_test)
# Bu da tercih edilebilir. Train test skorlar?? yak??n
# Sonu?? olarak k: 1,9,15,22 de??erlerini denedik. 
# Elbow a g??re en mant??kl??s?? 9 olarak g??r??n??yor
# Hesaplama maliyeti i??in 9 dan 22 ye ????kmaya gerek yok. Ancak 22 de makinan??z g????l??yse tercih edilebilir

######### Cross Validate For Optimal K Value
from sklearn.model_selection import cross_val_score, cross_validate
model = KNeighborsClassifier(n_neighbors=9)
scores = cross_validate(model, X_train_scaled, y_train, scoring = ['accuracy', 'precision','recall',
                                                                   'f1'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores

df_scores.mean()[2:]
#test_accuracy    0.924
#test_precision   0.923
#test_recall      0.929
#test_f1          0.926
#dtype: float64
# Burada skorumuz 0.923810 ????km????. k=9 iken 0.94 d?? accuracy de??eri. 
# Skorlar tutarl?? g??r??n??yor. Overfitting durumu da yok

######## Predict New Observation
# Model scaling yap??ld??????nda ve yap??lmad??????ndaki sonu??lar??n farkl??l??????n?? g??relim
new_observation = [[3.5, 4.3]]
knn_model.predict(new_observation) # Scale yap??lmadan ald??????m tahmin de kanser(1 s??n??f??na ait) dedi modelimiz
#array([1], dtype=int64)
#Tahminimiz 1 class' ina atandi :
#Asagida predict proba sonuclarina gore %60 oraninda 1 class' inin ciktigini goruyoruz. Bu yuzden model ornegimiz icin 1 class' ini secti 
#knn_model.predict_proba(new_observation)
# Bu tahmini 0.65 olas??l??kla yapt??
knn_model.predict_proba(new_observation)
#array([[0.34482759, 0.65517241]])
#Fakat yukaridaki islemimiz hatali. Cunku predict yapmak istedigimiz 
#sample' a scale islemi uygulamadik. Asagida ayni sample' a scale islemi
#uygulayarak modele verdigimizde 0 class' ina atama yapildigini goruyoruz.
#Data eger scale edildiyse predict edilecek data da scale edilmis olmali :
new_observation_scaled = scaler.transform(new_observation) # Scale edelim
new_observation_scaled
#array([[-1.1393583 , -0.62176572]])
knn_model.predict(new_observation_scaled)  # Scale yap??ld??ktan sonra ald??????m tahmin de kanser de??il(0 s??n??f??na ait) dedi modelimiz

knn_model.predict_proba(new_observation_scaled) # Bu tahmini 0.62 olas??l??kla yapt??

######### Gridsearch Method for Choosing Reasonable K Values
# Elbow a bakt??k. ??imdi di??er method olan gridsearch e bakal??m
from sklearn.model_selection import GridSearchCV
knn_grid = KNeighborsClassifier()
k_values= range(1,30)
param_grid = {"n_neighbors":k_values, "p": [1,2], "weights": ['uniform', "distance"]}
# En iyi parametreleri bulmak i??in paramatrelerimizde deneyece??i de??erleri tan??mlayal??m
knn_grid_model = GridSearchCV(knn_grid, param_grid, cv=10, scoring= 'accuracy')
knn_grid_model.fit(X_train_scaled, y_train)
knn_grid_model.best_params_  
#{'n_neighbors': 21, 'p': 1, 'weights': 'uniform'}
# k=21 , p:2(eucledian) , weight: uniform(noktalara e??it a????rl??klar versin)
# Burada grid search ??n verdi??i de??er genelde tercih edilmez
#GridSearch islemi en iyi degeri k=21 olarak secti. GridSearch en dusuk 
#error' u veren k degerini secer. Elbow metodda ise error' lara gore kendimiz 
#bir k degeri secebiliyorduk. Buyuk k degeri maliyet olarak geri donecegi icin
#Elbow metodu ile mutlaka error' lara bakip sonuca ona gore karar vermek gerekir.
#k=9 ile k=21 arasindaki skorlarda cok da bir fark yok. Bu yuzden k=9 degeri 
#ile modelimizi kurmaya karar verdik.
print('WITH K=21\n')
eval_metric(knn_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)
# n_neighbors=9,      test_accuracy: 94  with 54 error
# n_neighbors=21,    test_accuracy: 94   with 50 error
# Tercih noktas??nda sizden ne istenildi??ine g??re karar verilebilir
# Ancak hesaplama maliyetine g??re 9 u tercih etmek daha mant??kl?? g??r??n??yor

######### Evaluating ROC Curves and AUC
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve
knn_model = KNeighborsClassifier(n_neighbors=9).fit(X_train_scaled, y_train) # k=9 a g??re ...
plot_roc_curve(knn_model, X_test_scaled, y_test)

y_pred_proba = knn_model.predict_proba(X_test_scaled)
roc_auc_score(y_test, y_pred_proba[:,1])
# False positive rate lerin minimum olmas??n?? istiyoruz
# True positive rate i de maximize etmek istiyoruz
# AUC: Yakla????k 0.98 .. Kanser olma ve olmama durumunu y??zde 98 olarak ba??ar??l?? yap??yor diyebiliriz
#Elbow metodu baz alarak k=9 degerini sectik ve modelimizi kurduk. 
#Datamiz dengeli bir dataseti oldugu icin ROC/AUC grafigini cizdirdik. 
#Modelimizin genel performansi %98 :
########## Final Model and Model Deployment
import pickle
scaler = StandardScaler()
scaler.fit(X)  # T??m datay?? fit ediyoruz. Dikkat Fit_transform yapm??yoruz.
X_scaled = scaler.transform(X)
final_knn_model = KNeighborsClassifier(n_neighbors=9).fit(X_scaled,y)
pickle.dump(final_knn_model, open('knn_final.pkl', 'wb')) # wb: white binary(Daha az yer kaplamas?? i??in) 
# Modelimizi localimize kaydettik
pickle.dump(scaler, open('scaler_knn.pkl', 'wb')) # Yukar??da tan??mlad??????m??z scaler ?? da kaydediyoruz

########## Predict New Observations
# ??imdi kullanmak istedi??imizi d????enerek tekrar ??a????ral??m kaydettiklerimizi
loaded_scaler = pickle.load(open('scaler_knn.pkl', 'rb')) 
loaded_model = pickle.load(open('knn_final.pkl', 'rb'))
X.columns
X.describe()

# Yapay g??zlem olu??tural??m
new_obs = {"Gene One": [1, 3, 4.3, 5.6, 7, 9.5, 2, 6], 
           "Gene Two": [1, 4, 4, 5.5, 6.7, 10, 8, 1]
          }

samples = pd.DataFrame(new_obs)
samples

samples_scaled = loaded_scaler.transform(samples) # Buradaki datam?? d??n????t??r??yoruz. Localimize kaydetti??imiz scaler ile yap??yoruz bunu
samples_scaled

predictions = loaded_model.predict(samples_scaled) # k = 9 iken scale edilmi?? tahminlerimizi al??yoruz
predictions_proba = loaded_model.predict_proba(samples_scaled) # Burada da olas??l??klar??m??z??

samples["pred"] = predictions
samples["pred_proba_1"] = predictions_proba[:,1]  # 1 s??n??f??na ait olas??l??klar
samples["pred_proba_0"] = predictions_proba[:,0]  # 0 s??n??f??na ait olas??l??klar
samples
#   Gene One  Gene Two  pred  pred_proba_1  pred_proba_0
#0     1.000     1.000     0         0.000         1.000
#1     3.000     4.000     0         0.111         0.889
#2     4.300     4.000     1         1.000         0.000
#3     5.600     5.500     0         0.222         0.778
#4     7.000     6.700     0         0.000         1.000
#5     9.500    10.000     1         0.667         0.333
#6     2.000     8.000     0         0.000         1.000
#7     6.000     1.000     1         1.000         0.000
########## Pipeline
#What happens can be described as follows:

#Step 1: The data are split into TRAINING data and TEST data according to ratio 
#of train_test_split

#Step 2: the scaler is fitted on the TRAINING data

#Step 3: the scaler transforms TRAINING data

#Step 4: the models are fitted/trained using the transformed TRAINING data

#Step 5: the scaler is used to transform the TEST data

#Step 6: the trained models predict using the transformed TEST data

#Pipeline -----> fit ve transform ile yapilan islemleri siralandirir, optimize eder.

#Yani; scale, egitim ve tahmin islemlerinin hepsini tek bir kodla yapar.

#Yaptigimiz en buyuk hatalardan biri; X_train'i scale ettikten sonra ilerleyen
#islemlerde bunu unutup scale edilmemis datayi kullanmak. Pipeline bu sorunu
#ortadan kaldiriyor. Scale edilmesi gereken kisimlari scale eder, modele sokulmasi
#gereken kisimlari modele sokar.
#!!!!!!!! Pipeline, butun ML modellerinde kullanilabilir. !!!!!!!!!

# pipe.fit(X_train, y_train)--> scaler.fit_transform(X_train) --> knn.fit(scaled_X_train, y_train)
# pipe.predict(X_test) --> scaler.transform(X_test) --> knn.predict(scaled_X_test)


from sklearn.pipeline import Pipeline
operations = [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())] # Bunlar??n s??ras?? ??nemli. ??nce scaling sonra modelleme yapacak
# Data leakage olmamas?? i??in traine fit_transform yap??p, test e sadece transform yapacak. O ad??m?? halledecek
# Not olarak KNN burada default de??erle yani 5 de??eriyle devam ediyoruz burada pipeline ?? g??stermek ad??na
Pipeline(steps=operations)
#steps --------> Ben islemleri otomize edecegim ama hangi sirayla yapayim diye soruyor.
#Bu yuzden yukarida operation diye bir degisken tanimladik.
#operation icindekileri koseli parantezle belirtmek zorundayiz.
#'scaler', StandardScaler() ----------> scaler islemi yapacagim, StandardScaler() ile.
#'knn', KNeighborsClassifier() ------------> knn modelini kullanacagim.
#!!!!! operation icine yazdigimiz ilk islemler mutlaka fit ve transform yapan 
#islemler olmak zorunda. !!!!!!!
#!!!!! operation icine yazdigimiz ikinci islem mutlaka fit ve predict i
#slemi yapan algoritmalar, yani ML algoritmalari olmak zorunda. 
#(Sadece 1 tane model ismi yazilir. Birden fazlasi yazilamaz) !!!!!!
#Pipeline' ni bir degiskene atadigimizda bu pipeline hem bir scaler gibi hem de algoritma gibi davranmaya basliyor.
pipe_model = Pipeline(steps=operations)
#Bu degiskeni olusturduktan sonra fit islemini uyguladigimizda pipeline sunu yapar :
#pipe.fit(X_train, y_train)--> scaler.fit_transform(X_train) 
#--> knn.fit(scaled_X_train, y_train)
#Yani, once scale islemini sadece X_train'e uygular, sonra modeli kurar
#datayi X_train ve y_train'i egitir :
#Pipeline(steps=[('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])


pipe_model.fit(X_train, y_train) 
# E??itimi yapt??k 
#Egittigimiz pipeline'a predict islemini uyguladigimizda sunu yapar :
#pipe.predict(X_test) --> scaler.transform(X_test) --> knn.predict(scaled_X_test)
#Yani; X_test'i scale eder, sonra tahminlerini alir. 
#Yani artik X_testi scale ettim mi korkusu ortadan kalkar :
y_pred = pipe_model.predict(X_test)
y_pred

########## Model Performance
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
plot_confusion_matrix(pipe_model, X_test, y_test);
#Artik her yere model ismi olarak pipe_model' i verebiliriz.

print(classification_report(y_test, y_pred))

eval_metric(pipe_model, X_train, y_train, X_test, y_test)
# 65 errors


########## Changing the parameters of the pipe_model
pipe_model.get_params() 
# Burada nas??l yaz??ld?? ise pipe_model.set_params k??sm??nda o ??ekilde yazmal??y??z
#pipe_model icindeki parametreleri degistirmek istersek ne yapacagiz?
#Yukarida pipeline parametrelerine bakarsak; pipe_model icine tanimlanan 
#scaler ve modele sunu yapmis :
#operation icine tanimladigimiz 'scaler' ismini yazip iki tane alt cizgi 
#koymus ve sonra parametreleri yazmis (scaler__copy': True gibi)

#operation icine tanimladigimiz model olan 'knn' yi yazip iki tane alt 
#cizgi koymus ve sonra parametreleri yazmis (knn__n_neighbors': 5 gibi)

#Biz bu default degerleri degistirmek istersek sunu yapacagiz :
pipe_model.set_params(knn__n_neighbors= 9) 
# Parametreyi de??i??tiriyoruz burada. 5 i 9 ile de??i??tirdik
pipe_model.get_params() 
# Parametrenin de??i??ti??ini g??r??yoruz (knn__n_neighbors': 9 k??sm??nda)
pipe_model['scaler']        
# Datanin scale edilmis hali. Fit edilmeye hazir.

pipe_model["knn"]           
# Datanin egitilmis hali. Predict yapmaya hazir

############## GridSearch on Pipeline
from sklearn.model_selection import GridSearchCV
#Pipeline ile olusturulmus data GridSearch islemine tabi tutulacaksa, 
#GridSearch icine verilecek olan araliklarin pipe_model 
#parametrelerine uygun sekilde verilmesi gerekir, yani knn__n_neighbors gibi.
param_grid = {'knn__n_neighbors': range(1,30)} 
pipe_model = Pipeline(steps=operations)     
# Her yeni islemde modeli mutlaka sifirliyoruz.
pipe_grid = GridSearchCV(pipe_model, param_grid, cv=10, scoring= 'f1')
pipe_grid.fit(X_train,y_train) # E??itim
pipe_grid.best_params_
#{'knn__n_neighbors': 21}
############## CrossValidate on Pipeline
# CV ??ncesi modeli s??f??rlamam??z laz??m scaler ??m??z?? tan??mlad??k, n_neigbors=9 u tan??mlad??k tekrar 
#CrossValidate isleminde data her seferinde 10 parcaya bolunup 
#bir tanesi test datasi olarak ayrilir. Amaicerdeki bu test datasi da scale edilmis oldugu icin test datasi kopye ceker (Data Leakage). PipeLine kullandigimizda bunu onmelis oluruz.
operations = [('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=9))]
model = Pipeline(operations)
scores = cross_validate(model, X_train, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
#test_precision   0.923
#test_recall      0.931
#test_f1          0.927
#test_accuracy    0.925
# k=9 iken sonu??lar?? g??rm????t??k yukarda. Bunu pipeline ile de g??rm???? olduk
# Class chat soru: pipelineda parametrelerle oynamadan yukar??da de????kene atad??????m??z en iyi modeli kullanamaz m??y??z?
# Orion: Tekrar ayn?? parametreleri yaz??p e??itmeniz laz??m ([('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=9))])

########## Final pipe_model
operations = [('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=9))]
pipe_final = Pipeline(operations)
pipe_final.fit(X, y)

########## Predict New Observations with pipe_model
new_obs = {"Gene One": [1, 3, 4.3, 5.6, 7, 9.5, 2, 6],
           "Gene Two": [1, 4, 4, 5.5, 6.7, 10, 8, 1]
          }
samples = pd.DataFrame(new_obs)
samples
#Pipeline' dan once predict kisminda scale islemini yapmayi unutabiliyorduk. 
#Bu sorun da burda ortadan kalkiyor cunku pipeline bizim yerimize scale islemini otomatik yapiyor
predictions = pipe_final.predict(samples)
predictions

predictions_proba = pipe_final.predict_proba(samples)
predictions_proba

samples["pred"] = predictions
samples["pred_proba"] = predictions_proba[:,1]
samples

#################################################################################################################################
#%% LAB-3
# Data Set Information:
# Images of Kecimen and Besni raisin varieties grown in Turkey were obtained with CVS. A total of 900 raisin grains were used, including 450 pieces from both varieties. These images were subjected to various stages of pre-processing and 7 morphological features were extracted. These features have been classified using three different artificial intelligence techniques.
# Attribute Information:
# 1. Area: Gives the number of pixels within the boundaries of the raisin.
# 2. Perimeter: It measures the environment by calculating the distance between the boundaries of the raisin and the pixels around it.
# 3. MajorAxisLength: Gives the length of the main axis, which is the longest line that can be drawn on the raisin.
# 4. MinorAxisLength: Gives the length of the small axis, which is the shortest line that can be drawn on the raisin.
# 5. Eccentricity: It gives a measure of the eccentricity of the ellipse, which has the same moments as raisins.
# 6. ConvexArea: Gives the number of pixels of the smallest convex shell of the region formed by the raisin.
# 7. Extent: Gives the ratio of the region formed by the raisin to the total pixels in the bounding box.
# 8. Class: Kecimen and Besni raisin.
# 
# https://archive.ics.uci.edu/ml/datasets/Raisin+Dataset

# # Import libraries
# libraries for EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
import cufflinks as cf
#Enabling the offline mode for interactive plotting locally
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.io as pio
pio.renderers.default = "colab"
#To display the plots
get_ipython().run_line_magic('matplotlib', 'inline')

# sklearn library for machine learning algorithms, data preprocessing, and evaluation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, log_loss, recall_score, accuracy_score, precision_score, f1_score

# yellowbrick library for visualizing the model performance
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.cluster import KElbowVisualizer 

from sklearn.pipeline import Pipeline
# to get rid of the warnings
import warnings
warnings.filterwarnings("ignore")
warnings.warn("this will not show")

# !pip install cufflinks
# !pip install plotly

### Exploratory Data Analysis and Visualization
df = pd.read_excel("Raisin_Dataset.xlsx") # Reading the dataset
df.head() 
# T??rkiye de yeti??tirilen Ke??imen ve Besni kuru ??z??mlerinin 7 tane ??zelli??i verilmi??
# Bu ??z??mleri yap??sal ??zelliklerine g??re s??n??fland??raca????z
df.info() # Missing value yok. Dtype lar n??merik. Data cleaning yapmayaca????z
df.shape
df.duplicated().sum()   # No duplicates
df.isnull().sum().any() # No missing values
df.describe()    # Std > mean oldu??u durum yok gibi g??r??n??yor ??u an i??in

ax= sns.countplot(df["Class"]);
ax.bar_label(ax.containers[0]) # to show the proportion of each class
# Balanced bir data seti var elimizde.(Yani de??erlendirmek i??in accuracy yi kullanabiliriz)
# We have prety same amout of classes in the data set. So I can use accuracy as a metric to evaluate the performance of the classifier.

df["Class"] = df["Class"].map({"Kecimen":0,"Besni":1}) # mapping the classes to 0 and 1
# Ke??imen ve Besniyi say??sal de??erlere ??evirelim. Map ledik

df.iloc[:,:-1].iplot(kind="box") # Tek alanda t??m boxplotlar?? g??rd??k plotly ile

# S??n??f baz??nda bakal??m
fig = px.box(df, color="Class", color_discrete_map={"Kecimen":'#FF0000',"Besni":'#00FF00'})
fig.show()

df.iplot(kind="bar")
# Data ilk 450 sat??r Ke??imen e ait, sonraki 450 si Besni olarak s??n??fland??r??lm????
# Barplot a bakt??????m??z zaman 450 den sonras??nda areas?? b??y??kse besni(1) s??n??f??na ait diyebiliriz
# MajoraxisLength i??in ayn?? yorumu yapabiliriz
# MinorAxisLength i??in tam olarak ay??rt edici diyemeyebiliriz
# Eccentricity i??in de tam olarak ay??rt edici diyemeyiz
# ..
# Extent i??in de tam olarak ay??rt edici diyemeyiz
# Perimeter i??in b??y??k olursa besni s??n??f??na ait oldu??unu s??yleyebiliriz

fig = px.bar(df,x=df.index,y="Area",color="Class",color_discrete_map={"Kecimen":'#FF0000',"Besni":'#00FF00'})
fig.show()
# Area y?? inceledik class ile durumunu

plt.figure(figsize=(10,8))
sns.heatmap(df.select_dtypes(include='number').corr(),vmin=-1,vmax=1, annot=True, cmap='coolwarm')
# Ciddi korelasyonlar g??r??n??yor
# ??rne??in: Area ile Perimeter , Perimeter ile MajorAxislength
# Multicollinearity oldu??unu g??r??yoruz burada
# Multicollinearity da bask??n olan feature di??er feature ?? ezmi?? oluyor. Bask??l??yor bir nevi
# Bu sorunu bizim modelimiz ????zecek

corr_matrix = df.corr()
fig = px.imshow(corr_matrix)  # px: plotly.express k??t??phanesinden
fig.show()
# Bu ????kt??dan da ??e??itli insightlar elde edebiliriz
# Yukardaki heatmap e alternatif olarak kullanabiliriz

sns.pairplot(df, hue = "Class")
# S??n??flar??n genelde i?? i??e girdi??ini g??r??yoruz
# Yani datam??n birbirinden d??zg??n bir ??ekilde ayr??lmad??????n?? g??r??yoruz

fig = px.scatter_3d(df, x='Perimeter', y='Area', z='Extent',
              color='Class')
fig.show()
# Grafi??i tutup ??evirebiliriz
# ???? i??e giren noktalar oldu??unu g??r??yoruz.
# Bu da faydalanabilece??imiz ba??ka t??r bir grafik.
# EDA a??amas??ndan sonra modellemeye ge??i?? yapabiliriz

### Train | Test Split and Scaling
X=df.drop(["Class"], axis=1)
y=df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=10)

scaler =StandardScaler() # will be used in the pipelines
log_model = LogisticRegression() # will be used in the pipelines

log_pipe = Pipeline([("scaler",scaler),("log_model",log_model)]) # pipeline for logistic regression
# Operationlar??m??z?? yaz??yoruz SIRAYLA "scaler" ve "log_model"
log_pipe.fit(X_train, y_train) # Bunun i??inde scaling ve e??itim yap??l??yor

y_pred=log_pipe.predict(X_test)  # X_test scale i kendisi olu??turmu?? olacak ve tahmin alacak
y_pred_proba = log_pipe.predict_proba(X_test) # Ne kadar olas??l??kla tahmin ettik bunlar?? ald??k

### Model Performance
def eval_metric(model, X_train, y_train, X_test, y_test):
    """ to get the metrics for the model """
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

eval_metric(log_pipe, X_train, y_train, X_test, y_test) # to get the metrics for the model
# log_pipe: Pipeline ile olu??turdu??umuz modelimiz
# Balanced data setinde accuracy ye bak??p devam edebiliriz
# 0.87 test ve 0.87 train. Overfit g??r??nm??yor
# Bu skorlar?? CV ile kontrol edelim

#### Cross Validate
model = Pipeline([("scaler",scaler),("log_model",log_model)]) # Modeli tekrardan tan??ml??yorduk CV de
scores = cross_validate(model, X_train, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10,error_score="raise")
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
df_scores.mean()[2:]      # Accuracy: 0.86 # Alttaki ????kt?? ile kar????la??t??rd??????m??zda de??erler yak??n g??r??n??yoruz. Overfit durumu yok

eval_metric(log_pipe, X_train, y_train, X_test, y_test)

##### GridSearchCV
from sklearn.model_selection import GridSearchCV
# pipeline for logistic regression
model = Pipeline([("scaler",scaler),("log_model",log_model)]) # Grid search ??ncesi olu??turuyoruz
# l1: Lasso, l2: Ridge
penalty = ["l1", "l2"]                     # Multicollinearity yi ????zmek i??in 
# to get 20 values of C between -1 and 5
C = np.logspace(-1, 5, 20)                     # Hyperparameter
# balanced: class weights are balanced, None: no class weights
class_weight= ["balanced", None]               # Hyperparameter
# to get 4 values of solver
solver = ["lbfgs", "liblinear", "sag", "saga"]    # Optimize etmek i??in
# to get all the combinations of penalty, C, class_weight and solver
param_grid = {"log_model__penalty" : penalty,
              "log_model__C" : [C,1],
              "log_model__class_weight":class_weight,
              "log_model__solver":solver} 

# to get the best model
grid_model = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=10,
                          scoring = 'accuracy',       
                          n_jobs = -1) 

grid_model.fit(X_train,y_train) # E??itim yapt?? ve en iyi hyperparametreler ??zerinde
grid_model.best_params_ # to get the best parameters according to the best score

eval_metric(grid_model, X_train, y_train, X_test, y_test)  
# test set accuracy increased 0.87 to 0.88
# En iyi hyperparametrelerimiz ile sonu??lar??m??z
# Test scorum 0.88.. 1 puan iyile??mi?? oldu

#### ROC (Receiver Operating Curve) and AUC (Area Under Curve)
# Bunlar??n genel performans??n?? g??rmek i??in roc ??izdirelim
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score, auc, roc_curve, average_precision_score, precision_recall_curve
plot_roc_curve(grid_model, X_test, y_test) # we use ROC curve to get the AUC score and evaluate the model if it is good or not on every threshold
# Yorum Modelim %93 oran??nda ke??imen ve besni s??n??flar??n?? ayr????t??rabiliyor

plot_roc_curve(log_pipe, X_test, y_test)  # Eski modelim(Parametreleriyle oynamad??????m??z)
# ROC larda bir de??i??iklik olmad??
# log modeldeki ba??ar??m??z y??zde 88 bir de KNN e bakal??m

########## KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()  # to get a object of KNeighborsClassifier for pipeline 
# default k =5 olarak tan??mlad??k
knn_pipe = Pipeline([("scaler",scaler),("knn",knn)]) # pipeline for KNeighborsClassifier
knn_pipe.fit(X_train, y_train)  # Scale ve e??itim yapt??
knn_pred = knn_pipe.predict(X_test)
eval_metric(knn_pipe, X_train, y_train, X_test, y_test)
# Test accuracy 0.86, train accuracy.  0.88 k = 5 iken ba??ar??m %86. 10+16=26 tane yanl???? tahmin

##### Elbow Method for Choosing Reasonable K Values
# Optimal k y?? bulmak i??in kullan??yorduk
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
test_error_rates = []
for k in range(1,30):
    model = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=k))]) # p=1,weights="uniform",metric="minkowski"
    scores = cross_validate(model, X_train, y_train, scoring = ['accuracy'], cv = 10,error_score="raise")
    accuracy_mean = scores["test_accuracy"].mean() 
    test_error = 1 - accuracy_mean 
    test_error_rates.append(test_error)
# 1 den 30 a kadar k de??erlerini deneyece??iz
# Skorlar??n tutarl?? olmas?? a????s??nda cross_validate kullan??yoruz
# Tek seferlik ald??????m??z skordan ziyade 10 katl?? Cv ile test_error hesaplam???? olduk

# D??n yapt??????m??z gibi tek seferlik skorlara bakaca????z bir de(Yukardaki cv 10 katl?? skor, buras?? tek seferlik skor)
test_error_rates1 = []
for k in range(1,30):
    knn_model = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=k))])
    knn_model.fit(X_train,y_train) 
    y_pred_test = knn_model.predict(X_test)
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates1.append(test_error)

# ??stteki olu??turdu??umuz listelerin grafiklerini ??izdiriyoruz
plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates1, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.plot(range(1,30), test_error_rates, color='black', linestyle='-', marker='X',
         markerfacecolor='green', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')

# Cv ile yap??lan daha tutarl?? g??r??n??yor
# Minimum hata olan 26-27 yi se??ebiliriz. Ama maliyet artabilir
# 5,12,13,23 vs se??ilebilir
# Biz 5 i se??ece??iz. Hata daha y??ksek ama maliyet az(Ancak biz yine denemeler yapaca????z altta)

########### Scores by Various K Values
knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=2))])
knn.fit(X_train,y_train)
print('WITH K=2\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=2 i??in deneme yapt??k
# Test accuracy 0.84, train accuracy 0.92 .. Model overfit olmu?? oldu

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=8))])
knn.fit(X_train,y_train)
print('WITH K=8\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=8 i??in deneme yapt??k
# Test accuracy 0.83, train accuracy 0.88 .. 
# Ares Hoca: Skorlar birbirine yak??n gibi. Overfit olma ihtimali var..

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=25))])
knn.fit(X_train,y_train)
print('WITH 25K=\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=25 i??in deneme yapt??k
# Test accuracy 0.86, train accuracy 0.87 .. Skorlar yak??n tutarl?? model. Ancak iyile??me i??in bu kadar
# .. b??y??k k tercih edilir mi ? ...

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=5))])   
knn.fit(X_train,y_train)
print('WITH 5K=\n')
eval_metric(knn, X_train, y_train, X_test, y_test)
# k=5 i??in
# Test accuracy 0.86, train accuracy 0.88 . Model 10+16=26 tane hata yap??yor
# Elbow da k=5 se??meye karar verdik.
# Peki grid search ne diyecek k i??in bakaca????z
# Ondan ??nce Cross validation yapal??m(k=5 iken)


######### Cross Validate
model =Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=5))])
scores = cross_validate(model, X_train, y_train, scoring = ['precision','recall','f1','accuracy'], cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores
df_scores.mean()[2:]         # test Accuracy 0.85 .. Normalde 0.86 idi. Skorlar tutarl??

knn = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=5))])   # test_accuracy:    0.86 , train_accuracy :  0.88
                                                                                  # test_accuracy     0.85  (cross validation)
                                                                                  # (k=5 with elbow) with 26 wrong prediction
knn.fit(X_train,y_train)
print('WITH K=5\n')
eval_metric(knn, X_train, y_train, X_test, y_test)  

######### Gridsearch Method for Choosing Reasonable K Values
knn.get_params()
# pipeline for KNeighborsClassifier
knn_grid = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier())]) 
# to get all the values of k between 1 and 30
k_values= range(1,30)                  
# to get the values of weight
weight = ['uniform', 'distance']       # hyperparameter
# to get the values of p
p = [1,2]                              # hyperparameter
# to get the values of metric
metric = ['minkowski']                 # minkowski ye g??re p se??imi yap??lacak  
# to get all the combinations of k, weight, p and metric
param_grid = {'knn__n_neighbors': k_values,
              'knn__weights': weight, 
              'knn__p': p, 
              'knn__metric': metric} 
# to get the best model according to the best score
knn_grid_model = GridSearchCV(estimator= knn_grid, 
                             param_grid=param_grid,
                             cv=10, 
                             scoring= 'accuracy',
                             n_jobs=-1) 

knn_grid_model.fit(X_train, y_train)
knn_grid_model.best_params_ # to get the best parameters according to the best score
# 'uniform' : Hepsini e??it a????rl??kland??rd??
# k=14. Yukardaki grafikte 14 de hata en d??????k de??ildi ????nk?? farkl?? parametreler ile yapt??k
# Burada parametreler de??i??ti

test_error_rates2 = []
for k in range(1,30):
    model = Pipeline([("scaler",scaler),("knn",KNeighborsClassifier(n_neighbors=k, p=1))]) # p=1,weights="uniform",metric="minkowski"
    scores = cross_validate(model, X_train, y_train, scoring = ['accuracy'], cv = 10,error_score="raise")
    accuracy_mean = scores["test_accuracy"].mean() 
    test_error = 1 - accuracy_mean 
    test_error_rates.append(test_error)

plt.figure(figsize=(15,8))
plt.plot(range(1,30), test_error_rates1, color='blue', linestyle='--', marker='o',
         markerfacecolor='red', markersize=10)
plt.plot(range(1,30), test_error_rates, color='black', linestyle='-', marker='X',
         markerfacecolor='green', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K_values')
plt.ylabel('Error Rate')
# 14 te minimum oldu??unu g??r??yoruz

print('WITH K=14\n')      #  knn      test_accuracy :   0.85  (k=14 with gridsearch) with 27 wrong prediction

                          #  knn      test_accuracy :   0.86  (k=5 with elbow) with 26 wrong prediction
eval_metric(knn_grid_model, X_train, y_train, X_test, y_test)
# Skorlar birbirine yak??n ama tercihen k=5 se??iyoruz(Computational cost a????s??ndan)

############# Evaluating ROC Curves and AUC
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve, plot_roc_curve, roc_auc_score, roc_curve
model = KNeighborsClassifier(n_neighbors=14, p=1, metric="minkowski", weights="uniform") # best gridsearch model 
knn_model = Pipeline([("scaler",scaler),("knn",model)])
knn_model.fit(X_train, y_train)
# Bu kodlar?? roc curve ??izdirmek i??in yazt??k(grid search deki sonuca g??re)

# 0.85  (k=14 with gridsearch) with 27 wrong prediction
plot_roc_curve(knn_model, X_test, y_test) 

y_pred_proba = knn_model.predict_proba(X_test)
roc_auc_score(y_test, y_pred_proba[:,1])

model = KNeighborsClassifier(n_neighbors=5) # best elbow model
knn_model = Pipeline([("scaler",scaler),("knn",model)])
knn_model.fit(X_train, y_train)
# Bu kodlar?? roc curve ??izdirmek i??in yazt??k(elbow daki sonuca g??re)
#   knn test_accuracy :  0.85         (k=14 with gridsearch)       with 27 wrong prediction
#   knn test_accuracy :   0.86        (k=5 with elbow)             with 26 wrong prediction
plot_roc_curve(knn_model, X_test, y_test)

# k=14 iken Roc daha iyi buradaki k??s??m tercih meselesi
# k=5 ve k=14 .. ikiside do??ru model
# Ares Hoca: Ben 5 kom??ulu??u tercih ediyorum

# Log_modeli kurarken ??ok u??ra??mad??k
# KNN de u??ra??t??k, k de??erlerini denedik parametreleri denedik, elbow ve grid search e bakt??k vs...
# Bundan dolay??;
# Bu data i??in tercih olarak KNN ve log_model aras??nda log_modeli tercih ederiz
# KNN tercih edersem de k=5 derim
# logistic regression daha h??zl?? daha az maliyetli

# class chat soru: gridsearhcv de en iyi k de??eri 14 ????kt?? yani accuracy nin daha y??ksek olmas?? beklenmez miydi? neden k=5 de accuracy daha y??ksek ????kt??
# grid search de bir ??ok parametre ile oynad??k. Skorlar iyi ????kmad??
# Orion hoca:Trainden g??zel bir skor al??nabilir ama hold-out test setten iyi bir skor almam??z beklenir ama ????kmayabilir.
# .. ki neticede burada sonu?? ??yle olmu??(iyi skor olmam???? anlam??nda)

# # Conclusion 
# * log_model Accuracy Score: 0.88 
# * log_model AUC : 0.93           
# * knn Accuracy Score :   0.86  (k=5 with elbow)  - 0.85  (k=14 with gridsearch)
# * knn AUC : 0.88 (elbow) - 0.90 (gridsearch)
# * As a conclusion we aplied two models to predict raisins classes and we got prety decent scores both of them
# * We decided to use the Logistic Model because of its slightly better score than the knn models, 
#plus the interpretability of logistic regression and its lower computational cost.


