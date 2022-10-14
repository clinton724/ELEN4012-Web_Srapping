# ADR1_Scraping-library: Use of Scrapy over other scraping modules (Beautifulsoup, Selenium library)

## Overview
There are different python packages that can be used for web scraping and these are the scrapy library, selenium and beautiful soup. The 3 libraries have their pros and cons.

Scrapy is an open source collaborative framework for extracting the data from the websites what we need. Its performance is ridiculously fast and it is one of the most powerful libraries available out there

Selenium provides a way for the developer to write tests in a number of popular programming languages such as C#, Java, Python, Ruby, etc. his framework is developed to perform browser automation.

BeautfulSoup library can help us to pull the data out of HTML and XML files. But the problem with Beautiful Soup is it can’t able to do the entire job on its own. 

## Decision
We will use the scrapy library for crawling and extracting data from the target website.

## Status
Accepted

## Consequences

    ## Advantages
      Scrapy has built-in support for extracting data from HTML sources using XPath expression and CSS expression.

      It is a portable library i.e(written in Python and runs on Linux, Windows, Mac, and BSD)

      It consumes a lot less memory and CPU usage.

     ## Disadvantages
      Scrapy is single-threaded framework, you cannot use multiple threads within a spider at the same time. Hoyouver, you can create multiple spiders and piplines at the same time to make the process concurrent.

      Scrapy does not support multi-threading because it is built on Twisted, which is an Asynchronous http protocol framework.

