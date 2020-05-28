# coding:utf-8
# Copyright (C)
# Author: I
# Contact: 12157724@qq.com

boot_url = "http://www.nimrf.cugb.edu.cn/ept/treeList?zyglbm.id=3"
host = "http://www.nimrf.cugb.edu.cn/"
import scrapy

from spinder_rock.items import SpinderRockItem


class ItcastSpider(scrapy.Spider):
    name = "cugb"
    allowed_domains = ["www.nimrf.cugb.edu.cn"]
    start_urls = (
        boot_url,
    )

    def parse(self, response):
        yield from self.parse_rock_detail_(response)
        for i in range(1, 3701):
            yield from self.next_page_(i)

    def parse_rock_detail_(self, response):
        links = response.xpath("//*[@id='contentTable']/tbody/tr[*]/td[*]/a/@href").extract()
        for item in links:
            # print(item)
            yield scrapy.Request(url=host + item, callback=self.parse_rock_detail)

    def next_page_(self, page):
        print("next page ...", page)
        yield scrapy.FormRequest(url='http://www.nimrf.cugb.edu.cn/ept/treeList', callback=self.parse_rock_detail_,
                                 method='POST',
                                 formdata={
                                     "pageSize": "20",
                                     "pageNo": str(page),
                                     "zyglbm.id": "3"
                                 })

    def parse_rock_detail(self, response):
        print('parse_rock_detail:', response.url)
        id = response.xpath('//*[@id="contentTable"]/tr[2]/td[2]/span/text()').extract()
        name = response.xpath('//*[@id="contentTable"]/tr[3]/td[1]/span/text()').extract()
        img_src = response.xpath('//*[@id="tpzlTb"]/tr/td[2]/ul/li/a/@href').extract()
        item = SpinderRockItem()
        item['id'] = id
        item['name'] = name
        item['image_urls'] = [host+x for x in img_src]
        yield item

    def handle_next(self, response):
        print(response.text)
        self.parse_rock_detail(response)
