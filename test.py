from datetime import datetime
import requests
import csv
import bs4
#multi-threading imports
import concurrent.futures
from tqdm import tqdm
import schedule
import time


USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
REQUEST_HEADER = {
    'User-Agent': USER_AGENT,
    'Accept-Language':'en-US, en;q=0.5',
}

NO_THREADS = 5 #No of threads

def get_page_html(url):
    res = requests.get(url=url, headers=REQUEST_HEADER)
    return res.content

def get_stock_price(soup):
    price_div = soup.find('div', attrs = {'class': 'rPF6Lc'})
    price_divs = price_div.findAll('div', class_ = 'YMlKec fxKbKc')
    for div in price_divs:
        price = div.text.strip().replace('₹', '').replace(',','') # remove spaces
        # print(price)
        try:
            return float(price)
        except ValueError:
            print("Value Obtained for price could not be parsed")
            exit()

def get_stock_title(soup):
    title_main = soup.find('main')
    # print(title_main)
    title = title_main.find('div', attrs = {'class': 'zzDege'})
    l = list(title.text.strip().split())
    # print(l)
    return l[0]


def extract_info(url, output):
    stock_info = {}
    # print("\n\n")
    # print(f"Scraping URL:  {url}")
    html = get_page_html(url=url)
    try:
        soup = bs4.BeautifulSoup(html,'lxml')
        stock_info['title'] = get_stock_title(soup)
        stock_info['price'] = get_stock_price(soup)
        stock_info['date'] = datetime.today().strftime("%d-%m-%Y")
        stock_info['time'] = datetime.now().strftime("%H:%M:%S")
        # print(stock_info)
        # print("\n")
        output.append(stock_info)
    except:
        print("Exception")

def storeData(urls):
    stock_data = []
    data = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=NO_THREADS) as executor:
        for wkn in tqdm(range(0, len(urls))):
            executor.submit(extract_info, urls[wkn][0], stock_data)

    # output_file = 'output-{}.csv'.format(datetime.today().strftime("%m-%d-%Y"))
    # print(stock_data)
    data['Time'] = stock_data[0]['time']
    data['Date'] = stock_data[0]['date']

    for i in range(0, len(stock_data)):
        t = stock_data[i]['title']
        # print(t)

        match t:
            case "Bitcoin":
                data['BTC'] = stock_data[i]['price']
                with open('dataset-BTC.csv','a',newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerow(stock_data[i].values())

            case "Ether":
                data['ETH'] = stock_data[i]['price']
                with open('dataset-ETH.csv','a',newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerow(stock_data[i].values())

            case "Litecoin":
                data['LTC'] = stock_data[i]['price']
                with open('dataset-LTC.csv','a',newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerow(stock_data[i].values())

            case "Cardano":
                data['ADA'] = stock_data[i]['price']
                with open('dataset-ADA.csv','a',newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerow(stock_data[i].values())

            case "Tether":
                data['USDT'] = stock_data[i]['price']
                with open('dataset-USDT.csv','a',newline='') as outputfile:
                    writer = csv.writer(outputfile)
                    writer.writerow(stock_data[i].values())
            
            case _:
                print("Error while updating csv")

    # print(data)
    dataList = []
    dataList.append(data)
    cols = ['Date', 'Time', 'BTC', 'ETH', 'ADA', 'LTC', 'USDT']
    with open('dataset.csv','a',newline='') as outputfile:
        writer = csv.DictWriter(outputfile, fieldnames=cols)
        writer.writerows(dataList)
    
    print("Successfully updated csv file!! ")

if __name__ == "__main__":
    
    urls = []
    with open('crypto_urls.csv', newline='') as csvfile:
        urls = list(csv.reader(csvfile, delimiter=','))
        
    

    schedule.every(1).minutes.at(':58').until('23:59:00').do(storeData,urls)

    while 1:
        schedule.run_pending()
        time.sleep(1)
