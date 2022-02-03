from pathlib import Path


class Data:

    def __init__(self, data_dir, reverse=False):
        self.path = 'embeddings/'
        p = Path('./' + self.path)
        p.mkdir(exist_ok=True)
        self.ds_name = data_dir.split('/')[-1]

        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        #print(self.train_relations)

    def load_data(self, data_dir, data_type, reverse=False):
        with open(data_dir+"/"+data_type+ ".txt", "r") as f:
            data = f.read().strip().split("\n")
            if "nell" not in data_dir.lower():
                data = [i.split() for i in data]
            else:
                data = [i.split() for i in data]
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        data[i][j] = data[i][j].split(":")[-1]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        with open(self.path + self.ds_name + '_relations_str.txt', 'w') as filehandle:
            for listitem in relations:
                filehandle.write('%s\n' % listitem)
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        #print('@@@@@@@@',(entities[0]))
        # len 2102 for MT3k
        with open(self.path + self.ds_name + '_entities_str.txt', 'w') as filehandle:
            for listitem in entities:
                filehandle.write('%s\n' % listitem)
        return entities
