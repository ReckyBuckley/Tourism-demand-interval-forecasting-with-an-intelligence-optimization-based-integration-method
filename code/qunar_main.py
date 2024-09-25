# -*- coding: utf-8 -*-

import requests
import re
import random
import time
import pandas as pd
class Spiders():
    def __init__(self):
        self.oper=requests.session()
        self.first_url='https://travel.qunar.com/place/api/html/comments/poi/703517?poiList=true&sortField=0&rank=0&pageSize=10&page='                        
        self.first_headers={
            'referer': 'https://travel.qunar.com/p-oi703517-jiuzhaigoufengjingqu-0-1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
            'x-requested-with': 'XMLHttpRequest',
            }
    def Main_spiders(self):
        comments=[]
        dates=[]
        usernames=[]
        rates=[]
        for j in range(1,101):
            name_num=0
            date_num=0
            url=self.first_url+str(j)
            req=self.oper.get(url,headers=self.first_headers)
            all_data=req.text
            #Matching Comments
            pat_comment='<div class=\\\\"e_comment_content\\\\"><p class=\\\\"first\\\\">(.*?)</p>'
            comment=re.compile(pat_comment).findall(all_data)
            print(comment)
            for comment_i in range(len(comment)):
                comments.append(comment[comment_i])
            
            #Matching Date
            pat_date='<div class=\\\\"e_comment_add_info\\\\"><ul><li>(.*?)</li>'
            
            date=re.compile(pat_date).findall(all_data)
            
            for date_index in date:
                if len(date[date_num])>10:
                    date[date_num]=date[date_num][0:10]
                date_num+=1
            print(date)
            for date_i in range(len(date)):
                dates.append(date[date_i])
            #Matching Stars
            pat_rate='<span class=\\\\"cur_star star_(.*?)\\\\">'
            rate=re.compile(pat_rate).findall(all_data)      
            print(rate)
            for rate_i in range(len(rate)):
                rates.append(rate[rate_i])
            #Match User Name
            pat_username='<div class=\\\\"e_comment_usr_name\\\\">(.*?)&nbsp;<a rel=\\\\"nofollow\\\\" href=\\\\"javascript:;\\\\">(.*?)</a></div>'
            username=re.compile(pat_username).findall(all_data) 
            for name in username:
                username[name_num]=name[0]+'-'+name[1]
                #print(username[name_num])
            print(username)
            for username_i in range(len(username)):
                usernames.append(username[username_i])

            print('\n'+'-'*30+'Page'+str(j)+'crawl complete！'+'-'*30) 
            t=random.choice(range(1,3))
            print("Current delay%ds"%t)
            time.sleep(t) 
        all_results=pd.DataFrame({'date':dates,'comment':comments,'rate':rates})
        all_results.to_csv('main\data\Travel Website Reviews\Qunar data for Jiuzhaigou.csv',encoding='utf_8_sig',index=False)
if __name__ == '__main__':
    spiders=Spiders()
    spiders.Main_spiders()

    
    