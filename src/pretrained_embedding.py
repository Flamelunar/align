import numpy as np
import pickle

class pretrained_embed:
    def __init__(self):
        self.fname = None
        self.ext_words = []
        self.embeddings = None

    def create_dict_and_embedding(self, pretrained_file):
        self.fname = pretrained_file
        embedding_list = []
        with open(pretrained_file, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.strip().split()
                self.ext_words.append(values[0])
                vector = np.array(values[1:], dtype=np.float32)
                embedding_list.append(vector)
        self.embeddings = np.concatenate(embedding_list, axis=0)

    def save(self, dict_file_name, embedding_file_name):
        with open(dict_file_name, "w", encoding='utf-8') as fout_w, open(embedding_file_name, 'wb') as fout_e:
            fout_w.write("total-num=%d\n" % len(self.ext_words))
            for word in self.ext_words:
                fout_w.write("%s\t10\n" %(word))
            pickle.dump(self.embeddings, fout_e)

if __name__ == "__main__":
    pe = pretrained_embed()
    pe.create_dict_and_embedding("giga.100.txt")
    pe.save("extwords.txt", "giga.bin")
    embeds = pickle.load(open("giga.bin", 'rb'))
    print(np.sum(np.equal(embeds, pe.embeddings)))
    print(len(pe.ext_words))





