#!/usr/bin/python
import re
import urllib3
import csv
import os
import sys
import time
import datetime

import numpy as np
from bs4 import BeautifulSoup

# iterate all dates
#   iterate all tickers
#     repeatDowdload
#       save to ./input/data/news_date.csv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
class news_Reuters:
    def __init__(self):
        fin = open('./input/tickerList.csv')

        filterList = set()
        try: # this is used when we restart a task
            fList = open('./input/finished.reuters')
            for l in fList:
                filterList.add(l.strip())
        except: pass

        # https://uk.reuters.com/info/disclaimer
        # e.g. http://www.reuters.com/finance/stocks/company-news/BIDU.O?date=09262017
        self.suffix = {'AMEX': '.A', 'NASDAQ': '.O', 'NYSE': '.N'}
        self.repeat_times = 4
        self.sleep_times = 2
        self.iterate_by_day(fin, filterList)


    def iterate_by_day(self, fin, filterList):
        dateList = self.dateGenerator(1) # look back on the past X days
        for timestamp in dateList: # iterate all possible days
            print("%s%s%s" % (''.join(['-'] * 50), timestamp, ''.join(['-'] * 50)))
            self.iterate_by_ticker(fin, filterList, timestamp)

    def iterate_by_ticker(self, fin, filterList, timestamp):
        for line in fin: # iterate all possible tickers
            line = line.strip().split(',')
            ticker, name, exchange, MarketCap = line
            if ticker in filterList: continue
            print("%s - %s - %s - %s" % (ticker, name, exchange, MarketCap))
            self.repeatDownload(ticker, line, timestamp, exchange)

    def repeatDownload(self, ticker, line, timestamp, exchange):
        url = "https://www.reuters.com/finance/stocks/company-news/" + ticker + self.suffix[exchange]
        new_time = timestamp[4:] + timestamp[:4] # change 20151231 to 12312015 to match reuters format
        http = urllib3.PoolManager()
        for _ in range(self.repeat_times):
            try:
                time.sleep(np.random.poisson(self.sleep_times))
                response = http.request('GET',url + "?date=" + new_time)
                soup = BeautifulSoup(response.data, "lxml")
                hasNews = self.parser(soup, line, ticker, timestamp)
                if hasNews: return 1 # return if we get the news
                break # stop looping if the content is empty (no error)
            except: # repeat if http error appears
                print('Http error')
                continue
        return 0

    def parser(self, soup, line, ticker, timestamp):
        content = soup.find_all("div", {'class': ['topStory', 'feature']})
        if len(content) == 0: return 0
        fout = open('./input/dates/news_' + timestamp + '.csv', 'a+',encoding = 'utf-8')
        for i in range(len(content)):
            title = content[i].h2.get_text().replace(",", " ").replace("\n", " ")
            body = content[i].p.get_text().replace(",", " ").replace("\n", " ")

            if i == 0 and len(soup.find_all("div", class_="topStory")) > 0: news_type = 'topStory'
            else: news_type = 'normal'

            print(ticker, timestamp, title, news_type)
            fout.write(','.join([ticker, line[1], timestamp, title, body, news_type]) + '\n')
        fout.close()
        return 1

    def dateGenerator(self, numdays): # generate N days until now
        base = datetime.datetime.today()
        date_list = [base - datetime.timedelta(days=x) for x in range(0, numdays)]
        for i in range(len(date_list)): date_list[i] = date_list[i].strftime("%Y%m%d")
        return date_list

def main():
    news_Reuters()

if __name__ == "__main__":
    main()
