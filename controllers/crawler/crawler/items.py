# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CrawlerItem(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    rootURL = scrapy.Field()
    historicalData = scrapy.Field()
    market = scrapy.Field()

class ScraperItem(scrapy.Item):
    coin = scrapy.Field()
    Date = scrapy.Field()
    Market_cap = scrapy.Field()
    Volume = scrapy.Field()
    Open = scrapy.Field()
    Close = scrapy.Field()
    
