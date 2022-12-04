

import pandas as pd
df_=pd.read_csv("flo_data_20k.csv")
df=df_.copy()
pd.set_option("display.max_columns",None)
pd.set_option("display.width",500)
pd.set_option("display.float_format",lambda x: "%.4f" % x)

#aykırı değerleri baskılamak için treshold fonksiyonları tanımlandı.

def outlier_tresholds(dataframe,variable):
    quartile1=dataframe[variable].quantile(0.01)
    quartile3=dataframe[variable].quantile(0.99)
    interquantile_range=quartile3-quartile1
    up_limit=quartile3+1.5*interquantile_range
    low_limit=quartile1-1.5*interquantile_range
    return low_limit,up_limit


def replace_with_tresholds(dataframe,variable):
    low_limit,up_limit=outlier_tresholds(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=round(low_limit,0)
    dataframe.loc[(dataframe[variable]>up_limit),variable]=round(up_limit,0)

#aykırı değer baskılanması yapılıyor
replace_with_tresholds(df,"order_num_total_ever_online")
replace_with_tresholds(df,"order_num_total_ever_offline")
replace_with_tresholds(df,"customer_value_total_ever_offline")
replace_with_tresholds(df,"customer_value_total_ever_online")

#toplam alışveriş sayısı ve toplam harcamayı belirlemek
df["order_num_total"]=df["order_num_total_ever_online"]+df["order_num_total_ever_offline"]
df["customer_value_total"]=df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]


import datetime as dt
for i in df.columns:
    if "date" in i:
       df[i] = df[i].astype("datetime64[ns]")



df["last_order_date"].max()
today_date=dt.datetime(2021,6,1)
#adım2
df
#recency= her kullanıcı özelinde min ve max satın alma tarihi arasındaki fark(haftalık)
#frequency=tekrar eden toplam satın alma sayısı
#T:müşterinin yaşı.haftalık (analiz tarihinden ne kadar önce ilk satın alma yapılmıs)
#monetary value:satın alma başına ortalama kazanç

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
             "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).dt.days)/7,
             "T_weekly": ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7,
             "frequency": df["order_num_total"] ,
             "monetary_cltv_avg": df["customer_value_total"] / df["order_num_total"]})



#BG/NBD, Gamma-Gamma Modellerinin import edilmesi
!pip install lifetimes
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

#3 ay içerisinde beklenen satın almaların tahmin edilmesi
bgf=BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],cltv_df["recency_cltv_weekly"],cltv_df["T_weekly"])

cltv_df["expected_purchase_3_month"]=bgf.predict(12,cltv_df["frequency"],
                                                    cltv_df["recency_cltv_weekly"],
                                                     cltv_df["T_weekly"])

cltv_df["expected_purchase_6_month"]=bgf.predict(24,cltv_df["frequency"],
                                                    cltv_df["recency_cltv_weekly"],
                                                     cltv_df["T_weekly"])
cltv_df

#expected average value hesaplanması


ggf=GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])


cltv_df["expected_average_value"]=ggf.conditional_expected_average_profit(cltv_df["frequency"],cltv_df["monetary_cltv_avg"])

#6 aylık cltv hesaplanması

cltv=ggf.customer_lifetime_value(bgf,
                                 cltv_df["frequency"],
                                 cltv_df["recency_cltv_weekly"],
                                 cltv_df["T_weekly"],
                                 cltv_df["monetary_cltv_avg"],
                                 time=6 ,   #6aylık
                                 freq="W", #week olduğu
                                 discount_rate=0.01)
cltv.head()
cltv["customer_id"]=cltv_df["customer_id"]
cltv=cltv.reset_index()
cltv_final=cltv_df.merge(cltv,on="customer_id",how="left")
cltv_final.sort_values(by="clv",ascending=False).head(10)
cltv_final=cltv_final.drop("index",axis=1)
cltv_final=cltv_final.drop("level_0",axis=1)
#kısayol
#cltv_df["cltv"]=cltv
#CLTV SEGMENTATION

cltv_final["segment"]=pd.qcut(cltv_final["clv"],4,labels=["D","C","B","A"])

cltv_final.groupby("segment").agg({"count","mean","sum"})

