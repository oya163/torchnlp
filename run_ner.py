#!/home/osingh1/anaconda3/envs/py36/bin/python
"""
    Run ner based on torchnlp
"""

import os
from torchnlp import *
from torchnlp import ner
from torchnlp.common.hparams import HParams
from torchnlp.data.conll import conll2003_dataset
from torchnlp.data.nyt import nyt_ingredients_ner_dataset
from torchnlp.tasks.sequence_tagging import TransformerTagger
from torchnlp.tasks.sequence_tagging import BiLSTMTagger
from torchnlp.tasks.sequence_tagging import hparams_tagging_base
from torchnlp.tasks.sequence_tagging import train, evaluate, infer, interactive

from torchnlp.common.prefs import PREFS
from torchnlp.common.info import Info

import sys, os
import logging
import logging.config
from functools import partial

logger = logging.getLogger('main_log')

folder_name="nepsa_target"
log_path = folder_name+".log"
logging.basicConfig(filename=log_path,level=logging.INFO,filemode='w')

fh = logging.FileHandler(log_path)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

PREFS.defaults(
    data_train='train.txt',
    data_validation='val.txt',
    data_test='test.txt',
    early_stopping='highest_5_F1'
)

def main():
    
    data_path="./data"
    
    h1 = ner.hparams_lstm_ner()

    for i in range(0,5):
        kfold = i+1
        data_root=os.path.join(data_path, folder_name, str(kfold))

        ner.PREFS.prefs.update(data_root=data_root)

        model_name = folder_name+"_"+str(kfold)
        
        nyt_ingredients_ner = partial(nyt_ingredients_ner_dataset, root=PREFS.data_root)
        ner.train(model_name, BiLSTMTagger, nyt_ingredients_ner, hparams=h1)
        ner.evaluate(model_name, BiLSTMTagger, nyt_ingredients_ner, 'test')

if __name__=="__main__":
    main()
