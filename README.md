# YoshiNet
"YoshiNet" is a name of Deep Learning work collection for Yosiyoshi's works.

# 1, ddqn.py
Based on "PyTorch Intermediate Tutorials" on the website as below:


http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


The author modified it and remade it into a Double Deep-Q-Learning Neural Network.

Best result on: 
https://twitter.com/yosiyos38795255/status/984053427033403393


In the episode 15, the duration is longer than 1000.

# Description
Abstract of DDQN(Double Deep-Q-Network):


1, English


Deep Q-network is a reinforcement learnirng. This process is through trial and error, and optimize itself for maximization of score in the game, meaning the neural network coded in the technique of Deep Learning. Double DQN is a technique using the deepcopy of old learning data to feedback on the latest learning data regulary.


2, Chinese


Deep Q-networks是一种的强化学习。这是通过试错更新学习数据，最优化自己为了提升分数的算法，还是以深度学习的手法来所实装的神经网络之名。Double DQN以学习数据的deepcopy，把那个以前的学习数据定期反馈给神经网络，把它来覆盖在最新的学习数据。


Deep Q-Network源代码的来源是PyTorch的Documantation，我把它作为了基本。PyTorch的特征就是Define-by-run，那么我们在写代码的时候就觉得跟Chainer一样。重写转Double DQN（DDQN）写得很简单。不过那的结果，人工智能越来越进行学习，会学习整个游戏的特征，也会期待很高的分数出来。这还是并用深度学习和强化学习两手法的意义。


3, Japanese


Deep Q-networkは強化学習の一種。任意回数の試行錯誤を経て学習データを更新し、スコア最大化の目的で自らを最適化するアルゴリズム、または深層学習の手法で実装したニューラルネットワークの名称。Double DQNは、以前の学習データをディープコピーし、それを定期的に学習データへフィードバックする手法。


元となるDeep Q-networkの実装は、PyTorchのドキュメンテーションにあるコードをベースとする。PyTorchはDefine-by-runを特徴としており、ちょうどChainerと同様の感覚でコーディングできる。そして、Double DQN(DDQN)への書き換えは、コード量としては僅かで済む。しかし、その効果として、後半になればなるほど、AIはゲームの特徴を学習し、高いスコアを挙げることが期待される。これが深層学習と強化学習の手法を組み合わせる意義ではないだろうか。
