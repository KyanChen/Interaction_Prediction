import numpy as np
import json

file = '../../data/ppi_data.txt'
data = np.loadtxt(file, dtype=str, delimiter='\t')
class_num = [set(data[:, i]) for i in range(data.shape[1])]
input_gene = class_num[0] | class_num[1]
output_interaction = list(class_num[2])
interaction2id = {output_interaction[i]: i for i in range(len(output_interaction))}
json.dump(interaction2id, open('../../data/interaction2id.json', 'w'), indent=4)

input_gene = list(input_gene)
gene_vocab = {input_gene[i]: i for i in range(len(input_gene))}
json.dump(gene_vocab, open('../../data/gene_vocab.json', 'w'), indent=4)
