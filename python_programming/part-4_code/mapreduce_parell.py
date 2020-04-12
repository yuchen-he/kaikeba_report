import ray
from collections import Counter, defaultdict
import numpy as np
import heapq


MAX = 99 


def get_content(file, folder='/data'):
    file = folder + file
    f = open(file, 'r')
    return f.read()

@ray.remote
class Mapper():

    """
    1.读取数据 (wiki_01), words数据
    2.统计words数据的词频
    """

    def __init__(self, content_stream):
        self.content_stream = content_stream
        self.num_articles_processed = 0
        self.articles = []
        self.word_counts = []

    def get_new_content(self):
        try:
            contents = self.content_stream.next()
            print(" fetching..", contents[:10])
            self.word_counts.append(Counter(contents.split(" ")))
            self.num_articles_processed += 1
        except StopIteration:
            pass

    def get_range(self, article_index, keys):
        while self.num_articles_processed < article_index + 1:
            self.get_new_content()

        return [
            (k, v) for k, v in self.word_counts[article_index].items()
            if len(k) >= 1 and k[0] > keys[0] and k[0] <= keys[1]
        ]
        

@ray.remote
class Reducer():

    # 1.根据指定的keys（或者拆分规则) 从每一个Mapper中获取数据
    # 2.汇总词频

    def __init__(self, keys, *mappers):
        self.keys = keys
        self.mappers = mappers

    def reduce_result(self, article_index):
        word_count_sum = defaultdict(lambda: 0)

        count_ids = [mapper.get_range.remote(article_index, self.keys) for mapper in self.mappers]

        for count_id in count_ids:
            for k, v in ray.get(count_id):
                word_count_sum[k] += v
        print("Reduce counting...")

        return word_count_sum


class Stream():

    # 流式读取数据

    def __init__(self, max, folder):
        self.index = 0
        self.max = max
        self.folder = folder
        self.g = None

    def init(self):
        self.g = self.content()

    def file(self):
        return f"wiki_{0}{self.index}" if self.index < 10 else f"wiki_{self.index}"

    def content(self):
        while self.index < self.max:
            yield get_content(self.file(), self.folder)

    def next(self):
        if not self.g:
            self.init()
        return next(self.g)


if __name__ == '__main__':

    ray.init(redis_address='172.19.178.209:6379')

    num_of_mapper = 12
    num_of_reduce = 20


    streams = []
    folders = ['/data/AA/', '/data/AB/', '/data/AC/']

    for i in range(num_of_mapper):
        streams.append(Stream(MAX, folders[i % len(folders)]))

    chunks = np.array_split([chr(i) for i in range(ord("a"), ord("z") + 1)], num_of_reduce)

    keys = [(chunk[0], chunk[-1]) for chunk in chunks]


    mappers = [Mapper.remote(stream) for stream in streams]
    
    reducers = [Reducer.remote(key, *mappers) for key in keys]
    

    article_index = 0

    while True:
        print(" index=", article_index)
        total_counts = {}
        counts = ray.get([
            reducer.reduce_result.remote(article_index)
            for reducer in reducers
        ])

        for count in counts:
            total_counts.update(count)

        most_fre_words = heapq.nlargest(10, total_counts, key=total_counts.get)

        for word in most_fre_words:
            print(" ", word, total_counts[word])

        article_index += 1
