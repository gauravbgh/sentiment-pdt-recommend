import pandas as pd
import numpy as np
import random

item_based_reco = pd.read_csv('Dataset/item_based_reco.csv')
item_based_reco.set_index('reviews_username', inplace=True)

id_name_sent= pd.read_csv('Dataset/id_name_sentiment.csv')

class pdt_recommendation():
    
    def __init__(self):
        
        #self.username= username
        return None
    
    
    def recom_using_item_based(self, username): ##takes username as input and returns 20 reco pdts
        
        d = item_based_reco.loc[username].sort_values(ascending=False)[0:20]
        top_20 = list(d.index)
        
        return top_20
    
    def pdt_overall_sentiment(self, pdt_id): ##takes a pdt id as input serach the reviews and give the sentiment_score
        
        sentiment_score= id_name_sent.loc[id_name_sent['id']== pdt_id, 'overall_sentiment'].values[0]
        
        return sentiment_score
    
    def predict(self, username):
        
        top_20_pdt_id= self.recom_using_item_based(username)
        top_20_pdt_sent={}
        for id in top_20_pdt_id:
            top_20_pdt_sent[id] = self.pdt_overall_sentiment(id)
            
        top_20_pdt_sent= dict(sorted(top_20_pdt_sent.items(), key= (lambda x: -x[1])))

        top_5_pdt_id= list(top_20_pdt_sent.keys())[0:5]
        
        
        top_5_pdt_name= [id_name.loc[id_name['id']==x , 'name_with_brand'].values[0] for x in top_5_pdt_id]
        
        
        return(top_5_pdt_name)
            
        