# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 10:01:07 2018

@author: YosiYoshi

--Sample Result--
2018-08-20 10:51:42,036 : INFO : Training finished
Distance between English and French:  0.8327014489998941
Distance between English and German:  0.7756531718454136
Distance between English and Frisian:  0.6527180840556902
Distance between English and Welsh:  0.6815546899366858
Distance between English and Russian:  1.2578264975689721
Distance between English and Hungarian:  1.1659797321814762
Distance between English and Albanian:  1.2916865109191968
Distance between English and Greek:  1.0748241518305213
Distance between Russian and Latvian:  0.5425016168419532
"Distance" means "degree of semantic difference" in term of word2vector.
"""

from gensim.models.poincare import PoincareModel

data = [('Indo-European', 'Germanic'), ('Indo-European', 'Romance'),
        ('Indo-European', 'Celtic'), ('Indo-European', 'Greek'),
        ('Indo-European', 'Albanian'), ('Indo-European', 'Balto-Slavic'),
        ('Balto-Slavic', 'Slavic'), ('Balto-Slavic', 'Baltic'), ('Uralic', 'Finno-Ugric'),
        ('West Germanic', 'English'), ('West Germanic', 'German'), ('West Germanic', 'Dutch'),
        ('West Germanic', 'Frisian'), ('North Germanic', 'Danish'), ('North Germanic', 'Swedish'),
        ('North Germanic', 'Icelandic'), ('North Germanic', 'Faroese'), ('North Germanic', 'Norwegian'),
        ('West Romance', 'French'), ('East Romance', 'Romanian'), ('West Romance', 'Italian'),
        ('West Romance', 'Italian'), ('WestRomance', 'Sardinian'), ('West Romance', 'Catalan'),
        ('West Romance', 'Spanish'), ('West Romance', 'Portguese'), ('West Romance', 'Galician'),
        ('West Romance', 'Catalan'), ('West Romance', 'Provencal'), ('West Romance', 'Romansh'),
        ('Celtic', 'Breton'), ('Celtic', 'Welsh'), ('Celtic', 'Irish'),
        ('Baltic', 'Latvian'), ('Baltic', 'Lituanian'), ('Slavic', 'Russian'),
        ('Slavic', 'Polish'), ('Slavic', 'Ukrainian'), ('Slavic', 'Bulgagian'),
        ('Slavic', 'Czech'), ('Slavic', 'Slovakian'), ('Slavic', 'Croatian'),
        ('Slavic', 'Serbian'), ('Finno-Ugric', 'Finnish'), ('Finno-Ugric', 'Estonian'),
        ('Finno-Ugric', 'Hungarian'), ('Non-Indo-European', 'Uralic'), ('Romance', 'East-Romance'),
        ('Romance', 'West Romance'), ('Germanic','West Germanic'), ('Germanic', 'North Germanic')
        ,('Anglo-Norman', 'English'), ('Anglo-Norman', 'French'),  ('West Germanic', 'Scots'), ('West Romance', 'Anglo-Norman')]
model = PoincareModel(data, negative=2, size=2)
model.train(epochs=50)
print("Distance between English and French: ", model.kv.distance('English', 'French'))
print("Distance between English and German: ", model.kv.distance('English', 'German'))
print("Distance between English and Frisian: ", model.kv.distance('English', 'Frisian'))
print("Distance between English and Welsh: ", model.kv.distance('English', 'Welsh'))
print("Distance between English and Russian: ", model.kv.distance('English', 'Russian'))
print("Distance between English and Hungarian: ", model.kv.distance('English', 'Hungarian'))
print("Distance between English and Albanian: ", model.kv.distance('English', 'Albanian'))
print("Distance between English and Greek: ", model.kv.distance('English', 'Greek'))
print("Distance between Russian and Latvian: ", model.kv.distance('Russian', 'Latvian'))
