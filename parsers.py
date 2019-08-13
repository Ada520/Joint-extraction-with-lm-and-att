import sys
import os.path
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
sys.path.extend(['/mnt/f/PycharmProjects/multihead_joint_entity_relation_extraction-master', '/mnt/f/PycharmProjects/multihead_joint_entity_relation_extraction-master', '/mnt/d/Program Files/Python/Python36/python36.zip', '/mnt/d/Program Files/Python/Python36/DLLs', '/mnt/d/Program Files/Python/Python36/lib', '/mnt/d/Program Files/Python/Python36', '/mnt/d/Program Files/Python/Python36/lib/site-packages', '/mnt/c/Program Files/JetBrains/PyCharm 2018.3.2/helpers/pycharm_matplotlib_backend'])

import utils
import csv
import pandas as pd
import numpy as np

class headIdDoc:
    def __init__(self, id):
        self.docId = id
        self.token_ids = []
        self.tokens = []
        self.BIOs = []
        self.relations = []
        self.heads = []

        ###extend
        self.embedding_ids = []
        self.pos1_ids = []
        self.pos2_ids = []
        self.char_ids = []
        self.BIO_ids = []
        self.ecs = []
        self.ec_ids = []
        self.joint_ids = []

    def append(self, token_id, token, BIO, relations, heads):
        self.tokens.append(str(token))
        self.token_ids.append(token_id)
        self.BIOs.append(BIO)
        self.relations.append(relations)
        self.heads.append(heads)

    def extend(self,index, wordindices, dataset_set_characters,dataset_set_pos1,dataset_set_pos2 ,dataset_set_bio_tags, dataset_set_ec_tags,
               dataset_set_relations):
        for tId in range(len(self.tokens)):
            self.embedding_ids.append(int(utils.getEmbeddingId(self.tokens[tId], wordindices)))
            self.pos1_ids.append(dataset_set_pos1[index][tId] if tId < len(dataset_set_pos1[index]) else 122)
            self.pos2_ids.append(dataset_set_pos2[index][tId] if tId < len(dataset_set_pos2[index]) else 122)
            self.char_ids.append(utils.tokenToCharIds(self.tokens[tId], dataset_set_characters))
            self.BIO_ids.append(int(utils.getLabelId(self.BIOs[tId], dataset_set_bio_tags)))
            self.ecs.append(utils.getECfromBIO(self.BIOs[tId]))
            self.ec_ids.append(int(utils.getLabelId(utils.getECfromBIO(self.BIOs[tId]), dataset_set_ec_tags)))
            self.joint_ids.append(utils.getScoringMatrixHeads(self.relations[tId], dataset_set_relations, self.heads[tId]))


class headIdParser:
    def __init__(self, file):
        docNr = -1
        self.head_docs = []
        tokens = headIdDoc("")

        for i in range(file.shape[0]):
            if '#doc' in file[i][0] or i == file.shape[0] - 1:  # append all docs including the last one
                if (i == file.shape[0] - 1):  # append last line
                    tokens.append(int(file[i][0]), file[i][1], file[i][2],  utils.strToLst(file[i][3]),
                                  utils.
                                  strToLst(file[i][4]))  # append lines
                if (docNr != -1):
                    self.head_docs.append(tokens)
                docNr += 1
                tokens = headIdDoc(file[i][0])
            else:
                tokens.append(int(file[i][0]), file[i][1], file[i][2], utils.strToLst(file[i][3]),
                              utils.
                              strToLst(file[i][4]))  # append lines


def readHeadFile(headFile):
    # head_id_col_vector = ['tId', 'emId', "token", "nerId", "nerBilou","nerBIO", "ner", 'relLabels', "headIds", 'rels', 'relIds','scoringMatrixHeads','tokenWeights']
    head_id_col_vector = ['token_id', 'token', "BIO", "relation", 'head']
    headfile = pd.read_csv(headFile, names=head_id_col_vector, encoding="utf-8",
                           engine='python', sep="\t", quoting=csv.QUOTE_NONE).as_matrix()

    return headIdParser(headfile).head_docs

def preprocess(docs,wordindices,dataset_set_characters,dataset_set_pos1,dataset_set_pos2,dataset_set_bio_tags,dataset_set_ec_tags,dataset_set_relations):
    for i in range(len(docs)):
        docs[i].extend(i,wordindices,dataset_set_characters,dataset_set_pos1,dataset_set_pos2,dataset_set_bio_tags,dataset_set_ec_tags,dataset_set_relations)


class read_properties:
    def __init__(self,filepath, sep='=', comment_char='#'):
        """Read the file passed as parameter as a properties file."""
        self.props = {}
        #print filepath
        with open(filepath, "rt") as f:
            for line in f:
                #print line
                l = line.strip()
                if l and not l.startswith(comment_char):
                    key_value = l.split(sep)
                    self.props[key_value[0].strip()] = key_value[1].split("#")[0].strip('" \t')


    def getProperty(self,propertyName):
        return self.props.get(propertyName)

