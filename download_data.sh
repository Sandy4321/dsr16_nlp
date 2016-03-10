#!/usr/bin/env bash

cd data/
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
mkdir vqa
cd vqa
wget http://visualqa.org/data/mscoco/vqa/Questions_Train_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Val_mscoco.zip
wget http://visualqa.org/data/mscoco/vqa/Questions_Test_mscoco.zip
unzip *
