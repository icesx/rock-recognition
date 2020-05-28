# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    process = CrawlerProcess(get_project_settings())

    # 'followall' is the name of one of the spiders of the project.
    process.crawl('cugb')
    process.start()  # the script will block here until the crawling is finished