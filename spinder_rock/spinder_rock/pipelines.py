# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import scrapy
from scrapy.pipelines.images import ImagesPipeline


class SpinderRockImagePipeline(ImagesPipeline):

    def get_media_requests(self, item, info):
        for image_url in item['image_urls']:
            request = scrapy.Request(image_url)
            request.meta['item'] = item
            yield request

    def file_path(self, request, response=None, info=None):
        item = request.meta['item']
        open("rock.txt", "a").write(item['id'][0] + "," + item['name'][0]+"\n")
        return '/%s' % item['id'][0]+"_"+item['name'][0]+".jpg"
