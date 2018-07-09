# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 15:29:05 2018

@author: yosiyoshi
"""

import re
import numpy as np
import chainer
from chainer import Chain, optimizers, training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L

#There are sentences from two different topics(genres) of Chinese newspaper.
data = [
    ["人大 代表 追问 预算 3200 亿 其它 支 出去 哪 了? ", 1],
    ["这个 框 里 的 支出 金额 却 很 大 , 占 比 很 高.", 0],
    ["衙门 是 古代 官 署 的 一 种 俗称 , 就 是 官吏 办公 , 办事 的 地方.", 0],
    ["为什么 古代 官府 又 称为 衙门?", 1],
    ["我 的 眼里 为什么 常 含 泪水?", 1],
    ["北京 作为 城市 的 历史 可以 追溯 到 3 , 000 年前.", 0],
    ["贸易战 它 意味着 什么 呢?", 1],
    ["汉 元凤 元年 广阳郡 蓟县 属 幽州 管辖.", 0],
    ["何必 一边 造 贫困 , 一边 来 扶贫 呢?", 1],
    ["逼良为娼 的 社会 是 个 什么 社会?", 1],
    ["美国 撤出 亚洲 , 中国 就 能 成为 区域 老大?", 1],
    ["评估 结果 显示 , 9 年 来 , 各 省级 财政 透明度 总体 上 仍 不 及格.", 0],
    ["什么 有 了 人员 经费 还要 安排 劳务费 ?", 1],
    ["近期 , 卖 出 资产 的 大 佬 远 不止 解 直 锟一 人.", 0],
    ["中国 华信 董事局 主席 叶 简明 被 有关 部门 调查.", 0],
    ["魏 鹏 远 涉 案 3 亿 背后 谁 在 包庇?", 1],
    ["人民 在 问 钱 哪 去 了 ?", 1],
    ["业内人士 还 表示 , 中 植 系 喜欢 和 上市 公司 进行 资产 重组.", 0],
    ["西游记 里 有 几 妖怪?", 1],
    ["还有 多少 国人 在意 雷 洋 案 的 真相?", 1],
    ["东汉 光武 改制 时 ， 置 幽州 刺史部 于 蓟县.", 0],
    ["魏 则 西 案 暴露 出 的 医疗 连环 陷阱 是 谁 设 的?", 1],
    ["想 进 苹果 吗?", 1],
    ["爸爸 , 一个 人 越 有 钱 越 了不起 是 吗?", 1],
    ["隋 开皇 三 年 废除 燕郡.", 0],
    ["北上深天 价房 卖 给 谁 了?", 1],
    ["政府 不 是 解决 问题 的 办法.", 0],
    ["政府 恰恰 就是 问题 的 所在.", 0],
    ["中国 没有 价值连城 的 民族 品牌?", 1],
    ["北京 是 中国 大陆 最 重要 的 交通 枢纽.", 0],
    ["你 知道 有人 正在 掏 你 钱包 纳税 吗?", 1],
    ["全国 政协 副 主席 名单 现 多 个 变化.", 0],
    ["谁 可以 使 中国 经济 崩溃?", 1],
    ["为 垄断 个人 记忆 和 思想 叫好?", 1],
    ["贾 敬龙 已 死 ， 多少 问题 无 解.", 0],
    ["男人 为什么 不 想 回家?", 1],
    ["答案 是 经 媒体 曝光 的 约 16 座 .", 0],
    ["房地产 吊 半空 , 哪 有 回到 地面 上 踏实?", 1],
    ["物价 高 没 问题 ， 要 高 都 高 .", 0],
    ["这 完全 是 一厢情愿 的 愚蠢 思维.", 0],
    ["问题 在 畸形", 0],
    ["中国 到底 有 多少 座 白宫?", 1],
    ["美国 石油 是 国家 垄断 吗?", 1],
    ["这么 没 文化 怎么 敢 写 文章?", 1],
    ["雷 洋 死 了 . 他 的 死 在 中国 舆论 掀起 一 场 轩然大波.", 0],
    ["全篇 由 一百七十 多 个 问 句 所 组成.", 0],
    ["道德 赤贫 的 大国 还 能 走 多 远?", 1],
    ["北京 老城区 的 城市 道路 是 棋 盘式 的 格局 ， 横平 竖直 。.", 0],
    ["登高望远 勾 乡思.", 0],
    ["又 逢 重阳 登高 , 追索 生命 真 乡.", 0],
    ["九日 九日 重阳节 , 每逢 佳节 倍 思亲.", 0],
    ["智慧 从 哪里 来?", 1],
    ["这些 部门 为何 如此 惧怕 信息 公开?", 1],
    ["民主派 阵营 固守 行使 否决权 的 1 / 3 议席 ， 本土 派 掘 起 ， 香港 的 政治 面貌 进入 了 一个 新 阶段.", 0],
    ["香港 政治 版图 重 划 更 多 变化 将 到来?", 1],
    ["南北 方向 的 道路 有 中轴线 ， 以及 从 玉 蜓桥 到 雍和宫 的 东线 ， 和 开阳 桥 到 积水潭 桥 的 西线.", 0],
    ["什么 时候 请 喝 喜酒 ?", 1],
    ["中融 信托 在 资本 市场 这块 比较 猛 ， 买壳 装 资产 ， 是 资本 运作 的 高手.", 0],
    ["徐才 厚 到底 有 多少 财富 ， 恐怕 无 人 能 说 清 ， 他 自己 也 说不清.", 0],
    ["你 监控 人民 ， 人民 为何 不 能 拍 你 执法?", 1],
    ["北京 外围 的 城市 道路 则 是 环形 加 放射性 的 格.", 0],
    ["部长 大人 怒斥 外国 记者 “ 没 发言权 ” ， 只有 中国 人 自己 才 有 权 对 中国 的 人权 状况 发言.", 0]
#    ["Could you spare me a little of your time？", 1],

]

class GRU_SentenceClassifier(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, out_size):
        super(GRU_SentenceClassifier, self).__init__(
                xe = L.EmbedID(vocab_size, embed_size),
                eh = L.GRU(embed_size, hidden_size),
                eh2 = L.GRU(hidden_size, hidden_size),
                hh = L.Linear(hidden_size, hidden_size),
                hy = L.Linear(hidden_size, out_size)
        )
        
    def __call__(self, x):
        x = F.transpose_sequence(x)
        self.eh.reset_state()
        self.eh2.reset_state()
        for word in x:
            e = self.xe(word)
            h = self.eh(e)
            h2 = self.eh2(h)
        
        y = self.hy(h2)
        return y

N = len(data)
data_x, data_t = [], []
for d in data:
    data_x.append(d[0])
    data_t.append(d[1])
    
def sentence2words(sentence):
    stopwords = ["我", "个", "这", "与", "还", "如果", "是", "那", "又", 
                 "的","于", "在", "从", "到", "至", "之"]
    sentence = sentence.lower()
    sentence = sentence.replace("\n", "")
    sentence = re.sub(re.compile(r"[!-\/:-@[-`{-~]"), " ", sentence)
    sentence = sentence.split(" ")
    sentence_words = []
    for word in sentence:
        if (re.compile(r"^.*[0-9]+.*$").fullmatch(word) is not None):
            continue
        if word in stopwords:
            continue
        sentence_words.append(word)
    return sentence_words

words = {}
for sentence in data_x:
    sentence_words = sentence2words(sentence)
    for word in sentence_words:
        if word not in words:
            words[word] = len(words)

data_x_vec = []
for sentence in data_x:
    sentence_words = sentence2words(sentence)
    sentence_ids = []
    for word in sentence_words:
        sentence_ids.append(words[word])
    data_x_vec.append(sentence_ids)

max_sentence_size = 0
for sentence_vec in data_x_vec:
    if max_sentence_size < len(sentence_vec):
        max_sentence_size = len(sentence_vec)
for sentence_ids in data_x_vec:
    while len(sentence_ids) < max_sentence_size:
        sentence_ids.insert(0, -1)

data_x_vec = np.array(data_x_vec, dtype="int32")
data_t = np.array(data_t, dtype="int32")
dataset = []
for x, t in zip(data_x_vec, data_t):
    dataset.append((x, t))

EPOCH_NUM = 10
EMBED_SIZE = 100
HIDDEN_SIZE = 200
BATCH_SIZE = 5
OUT_SIZE = 2

model = L.Classifier(GRU_SentenceClassifier(
        vocab_size=len(words),
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        out_size=OUT_SIZE
))
optimizer = optimizers.Adam()
optimizer.setup(model)
train, test = chainer.datasets.split_dataset_random(dataset, N-10)
train_iter = chainer.iterators.SerialIterator(train, BATCH_SIZE)
test_iter = chainer.iterators.SerialIterator(test, BATCH_SIZE, repeat=False)
updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (EPOCH_NUM, "epoch"), out="result")
trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport(trigger=(1, "epoch")))
trainer.extend(extensions.PrintReport( ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"])) # エポック、学習損失、テスト損失、学習正解率、テスト正解率、経過時間
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.ParameterStatistics(model))
trainer.extend(extensions.dump_graph('main/loss'))
trainer.run()
