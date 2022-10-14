import scrapy
from ..items import CrawlerItem, ScraperItem
import sys
sys.path.insert(0, '../../../../')
from db import connection, cursor
from scrapy.contracts import Contract
from scrapy.exceptions import ContractFail
from scrapy.contracts.default import (
    UrlContract,
    ReturnsContract,
    ScrapesContract,
)

from itemadapter import is_item, ItemAdapter

class UrlSpider(scrapy.Spider):
    
    name = "getUrls"
    start_urls = ['https://www.coingecko.com']
    custom_settings = {'ITEM_PIPELINES': {'crawler.pipelines.CrawlerPipeline': 300}}
    
    def parse(self, response):
        """ Contract to check presence of fields in scraped items
        @url https://www.coingecko.com
        """
        for index in response.css("""body > div.container > div.gecko-table-container > div.coingecko-table 
                        > div.position-relative > div > table > tbody > tr:nth-child(n+1) > td.py-0.coin-name.cg-sticky-col.cg-sticky-third-col.px-0 
                        > div > div.tw-flex-auto > a::attr('href')"""):
             url = response.urljoin(index.get())
             yield scrapy.Request(url=url, callback=self.parseInnerPage)
        
        ##nextPage = response.css("body > div.container > div.gecko-table-container > div.coingecko-table > div.row.no-gutters.tw-flex.flex-column.flex-lg-row.tw-justify-center.mt-2 > nav > ul > li.page-item.next > a::attr('href')").get()    
        ##if nextPage is not None:
        ##   nextPage = response.urljoin(nextPage)
        ##  yield scrapy.Request(url=nextPage, callback=self.parse)
          

    def parseInnerPage(self, response):
       
       items = CrawlerItem()
       name = response.css("""body > div.container > div.tw-grid.tw-grid-cols-1.lg\:tw-grid-cols-3.tw-mb-4 > 
                    div.tw-col-span-3.md\:tw-col-span-2 > div > div.tw-col-span-2.md\:tw-col-span-2 > 
                    div.tw-flex.tw-text-gray-900.dark\:tw-text-white.tw-mt-2.tw-items-center > div::text""").get()
       historicalData_raw = response.css("#navigationTab > li:nth-child(4) > a::attr('href')").get()
       historicalData = response.urljoin(historicalData_raw)
       market_raw = response.css("#navigationTabMarketsChoice::attr('href')").get()
       market = response.urljoin(market_raw)
       rootURL = response.url
       temp = name.split("\n")
       name = temp[1]
       items['name'] = name
       items['rootURL'] = rootURL
       items['historicalData'] = historicalData
       items['market'] = market
       yield items

class getData(scrapy.Spider):
    name = "rawData"
    custom_settings = {'ITEM_PIPELINES': {'crawler.pipelines.ScraperPipeline': 300}}
    cursor.execute("select Cryptocurrency, historicalData_URL from urlMapping")
    data = cursor.fetchall()
    connection.commit()

    def start_requests(self):
            for index in self.data:
                yield scrapy.Request(url=index[1], callback=self.parse)
    
    def parse(self, response):
           items = ScraperItem()
           name = response.css("""body > div.container > div.tw-grid.tw-grid-cols-1.lg\:tw-grid-cols-3.tw-mb-4 > 
                    div.tw-col-span-3.md\:tw-col-span-2 > div > div.tw-col-span-2.md\:tw-col-span-2 > 
                    div.tw-flex.tw-text-gray-900.dark\:tw-text-white.tw-mt-2.tw-items-center > div::text""").get()
           rows = response.css("body > div.container > div.card-body > div > div > table > tbody > tr")
           
           for index in rows:
                Date = index.css("th::text").extract()
                Market_cap = index.css("td:nth-child(2)::text").extract()
                Volume = index.css("td:nth-child(3)::text").extract()
                Open = index.css("td:nth-child(4)::text").extract()
                Close = index.css("td:nth-child(5)::text").extract()
                temp1 = Market_cap[0].split("\n")
                temp2 = Volume[0].split("\n")
                temp3 = Open[0].split("\n")
                temp4 = Close[0].split("\n")
                temp0 = name.split("\n")
                coin = temp0[1]
                Date = Date[0]
                Market_cap = temp1[1]
                Volume = temp2[1]
                Open = temp3[1]
                Close = temp4[1]
                items['coin'] = coin
                items['Market_cap'] = Market_cap
                items['Date'] = Date
                items['Volume'] = Volume
                items['Open'] = Open
                items['Close'] = Close
                yield items





       
