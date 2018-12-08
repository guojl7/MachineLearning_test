#-*- coding:utf-8 -*-
import twitter
from time import sleep
import re
import FPgrowth

def textParse(bigString):
    urlsRemoved = re.sub('(http:[/][/]|www.)([a-z]|[A-Z]|[0-9]|[/.]|[~])*', '', bigString)    
    listOfTokens = re.split(r'\W*', urlsRemoved)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def getLotsOfTweets(searchStr):
    CONSUMER_KEY = 'qq'
    CONSUMER_SECRET = 'qq'
    ACCESS_TOKEN_KEY = 'qq'
    ACCESS_TOKEN_SECRET = 'qq'
    api = twitter.api(consumer_key=CONSUMER_KEY, 
                      consumer_secret=CONSUMER_SECRET,
                      access_token_key=ACCESS_TOKEN_KEY, 
                      access_token_secret=ACCESS_TOKEN_SECRET)
    #you can get 1500 results 15 pages * 100 per page
    resultsPages = []
    for i in range(1,15):
        print "fetching page %d" % i
        searchResults = api.GetSearch(searchStr, per_page=100, page=i)
        resultsPages.append(searchResults)
        sleep(6)
    return resultsPages

def mineTweets(tweetArr, minSup=5):
    parsedList = []
    for i in range(14):
        for j in range(100):
            parsedList.append(textParse(tweetArr[i][j].text))
    initSet = FPgrowth.createInitSet(parsedList)
    myFPtree, myHeaderTab = FPgrowth.createTree(initSet, minSup)
    myFreqList = []
    FPgrowth.mineTree(myFPtree, myHeaderTab, minSup, set([]), myFreqList)
    return myFreqList

if __name__ == '__main__':
    lotsOtweets = getLotsOfTweets('RIMM')
    listOfTerms = mineTweets(lotsOtweets, 20)
    print(listOfTerms)
    
