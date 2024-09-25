import requests
import json
import pandas as pd
from tqdm import tqdm
import time

# Create a result list and add a header
result1 = [['Date', 'Text', 'Likes', 'Stars']]

total_pages = 300

for pagen in tqdm(range(0, total_pages), desc='crawl progress', unit='page'):
    
    payload = {
        "arg": {
            "channelType": 2,
            "collapseTpte": 0,
            "commentTagId": 0,
            "pageIndex": pagen,
            "pageSize": 10,
            "poiId": 89895,
            "sourseType": 1,
            "sortType": 9,
            "starType": 0
        },
        "head": {
            "cid": "09031158219828278574",
            "ctok": "",
            "cver": "1.0",
            "lang": "01",
            "sid": "8888",
            "syscode": "09",
            "auth": "",
            "xsid": "",
            "extension": []
        }
    }

    
    postUrl = "https://m.ctrip.com/restapi/soa2/13444/json/getCommentCollapseList"

    html = requests.post(postUrl, data=json.dumps(payload)).text
    html_1 = json.loads(html)

     # Check if 'items' is present in the response and is not None
    if 'items' in html_1["result"] and html_1["result"]["items"] is not None:
        commentItems = html_1["result"]["items"]
        for i in range(len(commentItems)):
            # Check if commentItems[i] is None
            if commentItems[i] is not None:
                if 'content' in commentItems[i] and 'publishTypeTag' in commentItems[i] and 'replyCount' in commentItems[i] and 'score' in commentItems[i]:
                    commentDate = time.strftime("%Y/%m/%d",time.localtime(int(commentItems[i]['publishTime'][6:16])))
                    commentDetail = commentItems[i]['content']
                    likeCount = commentItems[i]['replyCount']
                    starScore = commentItems[i]['score']

                    # Append data to the result list
                    result1.append([commentDate, commentDetail, likeCount, starScore])
    else:
        print(f"Page {pagen} has no items or items is None")
    time.sleep(1)  

# Creating a DataFrame
df = pd.DataFrame(result1[1:], columns=result1[0])

df.to_excel('main\data\Travel Website Reviews\Crtip data for Jiuzhaigou.xlsx', index=False, encoding='utf-8')
