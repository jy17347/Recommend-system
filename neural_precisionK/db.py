from pymongo.mongo_client import MongoClient
import pandas as pd

# pandas to mongodb
def pd_to_dic(df):
    return df.to_dict('records')

# mongodb to pandas
def dic_to_pd(dic):
    return pd.DataFrame.from_records(dic)

def get_user():
    client = MongoClient("mongodb+srv://KWIX:KWIX1234!@cluster0.sqcy3o3.mongodb.net/?retryWrites=true&w=majority")
    userInfo=client.KWIX.recommendData
    rows=userInfo.find()
    data=[]
    for row in rows:
        data.append(row)

    pd=dic_to_pd(data)

    data = pd[['id','height','weight', 'sex','age','bmi','during']].to_numpy()

    return data