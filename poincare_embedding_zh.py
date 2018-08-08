"""
Created on Tue Aug  7 10:01:07 2018

@author: YosiYoshi（李善）

Poincare embedding algorithm about Bilibili恶搞(MAD).
"""

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath

data = [('诸葛孔明', '村夫'), ('王朗', '无耻之人'), ('鲁迅', '跟我有什么关系'),
             ('诸葛孔明', '丞相'), ('周树人', '跟我有什么关系'), ('刘醒', '吔屎你'),
             ('Tony哥', '吔屎你'), ('王朗', '司徒'), ('王朗', '嘤嘤狂吠')]
model = PoincareModel(data, negative=2, size=2)
model.train(epochs=50)
print("Distance between 诸葛孔明 and 王朗: ", model.kv.distance('诸葛孔明', '王朗'))
print("Distance between 鲁迅 and 周树人: ", model.kv.distance('鲁迅', '周树人'))
print("Distance between 刘醒 and Tony哥: ", model.kv.distance('刘醒', 'Tony哥'))
