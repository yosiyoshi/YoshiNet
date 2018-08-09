# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:01:07 2018

@author: YosiYoshi

--Sample Result--
Distance between Cognitive Science and Natural Language Processing:  1.1503508546374988
Distance between AI and Natural Language Processing:  0.681424686195149
Distance between AI and Philosophy:  0.02879856608753927

"Distance" means "degree of semantic difference" in term of word2vector.
"""

from gensim.models.poincare import PoincareModel, PoincareRelations
from gensim.test.utils import datapath

data = [('cognitive science', 'linguistics'), ('cognitive science', 'education'),
        ('cognitive science', 'neuroscience'), ('cognitive science', 'AI'),
        ('cognitive science', 'anthropology'), ('cognitive science', 'psychology'),
        ('cognitive science', 'philosophy'), ('NLP', 'computer science'),
        ('NLP', 'AI'), ('NLP', 'linguistics'), ('NLP', 'philosophy'),
        ('linguistics', 'computational linguistics'),
        ('computational linguistics', 'computer science')]
model = PoincareModel(data, negative=2, size=2)
model.train(epochs=50)
print("Distance between Cognitive Science and Natural Language Processing: ", model.kv.distance('cognitive science', 'NLP'))
print("Distance between AI and Natural Language Processing: ", model.kv.distance('AI', 'NLP'))
print("Distance between AI and Philosophy: ", model.kv.distance('AI', 'philosophy'))
