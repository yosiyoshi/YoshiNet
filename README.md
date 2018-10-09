# YoshiNet
"YoshiNet" is a name of Deep Learning work collection for Yosiyoshi's works.

# Requirements
Software

/Python 3.x

/TensorFlow

/Chainer

/PyTorch

/CatBoost

/gensim

Knowledge

/TensorFlow

and

/PyTorch Documentation

Almost based on them, but I added some functions.

# 1, ddqn/ddqn2.py (2 code files)
Based on "PyTorch Intermediate Tutorials" on the website as below:


http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


The author modified it and remade it into a Double Deep-Q-Learning Neural Network.

Best result on: 
https://twitter.com/yosiyos38795255/status/984053427033403393

The same picture is shown as "highestscore.png" of ddqn.py("Koishi") scoring.

In the episode 15, the duration is longer than 1000.

The competition between ddqn.py("Koishi") and ddqn2.py("Satori") is shown on competition.png .

# 2, mnist_recog.py
Highest score of

"step 1700, lr=0.01 ,training accuracy 0.9903"

and requires an image named "input_number.jpg" as default.

Best Result on:
https://twitter.com/yosiyos38795255/status/985522879017070592

# 3, gru_zh.py

Best result on:
https://twitter.com/yosiyos38795255/status/1016291441356636160

# 4, poincare_embedding_zh/_en.py (2 code files)

Sample code "poincare_embedding_zh.py" result on:
https://twitter.com/yosiyos38795255/status/1026995495321759744

# 5, cbcls.py

Tutorial code for CatBoost(Yandex) array-like data prediction of Machine Learning.

# 6, poincare_embedding_jp.py "大学東方夏の自由研究"

東方カップリングの階層表現学習におけるPoincaré Embeddingsの応用

Poincaré Embeddings for Learning Hierarchical Representations of Touhou Coupling 

@Yosiyoshi: 南秘

KW: Touhou Project, Poincaré Embeddings, Hierarchical Representations, Machine Learning

東方projectのキャラクターは公式設定、および二次創作によりカップリングが複数存在する。
[M Nickel and D Kiela, 2017]による
"Poincaré Embeddings for Learning Hierarchical Representations"
の触発を受け、Poincaré Embeddingsを東方カップリングの階層表現学習に応用した。

## 実験方法

[Niconico大百科, 2018]の"東方projectのカップリング一覧:キャラクター別"
(http://dic.nicovideo.jp/a/%E6%9D%B1%E6%96%B9project%E3%81%AE%E3%82%AB%E3%83%83%E3%83%97%E3%83%AA%E3%83%B3%E3%82%B0%E4%B8%80%E8%A6%A7)

より、n=['博麗霊夢','霧雨魔理沙','東風谷早苗','アリス・マーガトロイド']の四名に係るカップリングを
任意に抽出した。カップリング相手をm=[任意の文字列]とし、カップリングをdata=(n,m)と定義する。

dataをgensim.models.poincare.PoincareModelに入力し、epoch=50で学習する。
学習結果はPoincaré双曲面上にプロットされる。

## 結果

れいまりの距離: 0.9867836799596653

まりありの距離: 1.193140088539443

れいありの距離: 1.33561024791922

れいれみの距離: 0.8767354255212357

れいさとの距離: 0.5425000174999931

まりれみの距離: 0.22278990408263546

まりさとの距離: 0.4602781802364823

さなれみの距離: 0.18275833633037145

さなさとの距離: 0.39465960366370423

ありれみの距離: 0.9773531395801837

ありさとの距離: 1.0359620245638737

## 結論

上記結果より、れいさと（霊夢・さとり）、まりれみ（魔理沙・レミリア）、まりさと（魔理沙・さとり）、
さなれみ（早苗・レミリア）、さなさと（早苗・さとり）のカップリング両者間における距離
が特に小さい。したがって、レミリア・スカーレットと古明地さとりは上記四名とカップリングにおいて親和的である。

# Description for 1, ddqn.py
Abstract of DDQN(Double Deep-Q-Network):


1, English


Deep Q-network is a reinforcement learnirng. This process is through trial and error, and optimize itself for maximization of score in the game, meaning the neural network coded in the technique of Deep Learning. Double DQN is a technique using the deepcopy of old learning data to feedback on the latest learning data regulary.


2, Chinese


Deep Q-networks是一种的强化学习。这是通过试错更新学习数据，最优化自己为了提升分数的算法，还是以深度学习的手法来所实装的神经网络之名。Double DQN以学习数据的deepcopy，把那个以前的学习数据定期反馈给神经网络，把它来覆盖在最新的学习数据。


Deep Q-Network源代码的来源是PyTorch的Documantation，我把它作为了基本。PyTorch的特征就是Define-by-run，那么我们在写代码的时候就觉得跟Chainer一样。重写转Double DQN（DDQN）写得很简单。不过那的结果，人工智能越来越进行学习，会学习整个游戏的特征，也会期待很高的分数出来。这还是并用深度学习和强化学习两手法的意义。


3, Japanese


Deep Q-networkは強化学習の一種。任意回数の試行錯誤を経て学習データを更新し、スコア最大化の目的で自らを最適化するアルゴリズム、または深層学習の手法で実装したニューラルネットワークの名称。Double DQNは、以前の学習データをディープコピーし、それを定期的に学習データへフィードバックする手法。


元となるDeep Q-networkの実装は、PyTorchのドキュメンテーションにあるコードをベースとする。PyTorchはDefine-by-runを特徴としており、ちょうどChainerと同様の感覚でコーディングできる。そして、Double DQN(DDQN)への書き換えは、コード量としては僅かで済む。しかし、その効果として、後半になればなるほど、AIはゲームの特徴を学習し、高いスコアを挙げることが期待される。これが深層学習と強化学習の手法を組み合わせる意義ではないだろうか。
