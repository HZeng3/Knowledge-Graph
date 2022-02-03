from load_data import Data
import numpy as np
import torch
import time
from collections import defaultdict
from model import *
from torch.optim.lr_scheduler import ExponentialLR
import argparse
from tqdm import tqdm
import pickle
from pathlib import Path

    
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., cuda=False, 
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,
                 label_smoothing=0.):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    
    def evaluate(self, model, data, verbose):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        # print("Number of data points: %d" % len(test_data_idxs))
        
        #for i in tqdm(range(0, len(test_data_idxs), self.batch_size)):
        for i in range(0, len(test_data_idxs), self.batch_size):
            #print(f'len(test_data_idxs): {len(test_data_idxs)}')
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            '''
            if i == 0:
                print(f'e1_idx: {e1_idx}')
                print(f'{e1_idx.dtype}')
                print(f'r_idx: {r_idx}')
                print(f'e2_idx: {e2_idx}')
                
                for key in self.entity_idxs.keys():
                    value = self.entity_idxs[key]
                    if value in e1_idx:
                        print(f"found in e1 idx: key: {key} and value: {value}")
                    
                    if value in e2_idx:
                        print(f"found in e2 idx: key: {key} and value: {value}")
            '''
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            '''if i == 0:
                predictions = model.forward(e1_idx, r_idx, True)
            else:'''
            predictions = model.forward(e1_idx, r_idx)
            # print(f'shape of prediction [75]: {predictions.shape}')
            for j in range(data_batch.shape[0]):
                filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j,e2_idx[j]].item()
                predictions[j, filt] = 0.0
                predictions[j, e2_idx[j]] = target_value

            sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
                ranks.append(rank+1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
        if verbose:
            print('Hits @10: {0}'.format(np.mean(hits[9])))
            print('Hits @3: {0}'.format(np.mean(hits[2])))
            print('Hits @1: {0}'.format(np.mean(hits[0])))
            print('Mean rank: {0}'.format(np.mean(ranks)))
            print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))




    def train_and_eval(self, verbose_iter):
        print("Training the TuckER model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

        path = 'embeddings/'
        p = Path('./' + path)
        p.mkdir(exist_ok=True)

        pickle.dump(self.entity_idxs, open(path + d.ds_name + '_entity_idxs.pkl', 'wb'))
        pickle.dump(self.relation_idxs, open(path + d.ds_name + '_relation_idxs.pkl', 'wb'))

        # print(f"self.relation_idxs: {self.relation_idxs}")
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        #print(f"er_vocab: {er_vocab}")
        #print(f"er_vocab_pairs: {er_vocab_pairs}")
        print("Starting training...")
        verbose = False
        for it in tqdm(range(1, self.num_iterations+1)):
        #for it in range(1, self.num_iterations+1):
            
            start_train = time.time()
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                r_idx = torch.tensor(data_batch[:,1])  
                #print(f"e1_idx: {e1_idx} and r_idx: {r_idx}")
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                predictions = model.forward(e1_idx, r_idx)
                # if it == self.num_iterations:
                #     print(f'predictions.shape [142]: {predictions.shape}')
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))           
                loss = model.loss(predictions, targets)
                loss.backward()

                opt.step()
                
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()
            #print(it)
            # if it > verbose_iter:
            #     verbose = True 
            #print(time.time()-start_train)    
            #print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if verbose:
                    print("Validation:")
                if it > verbose_iter:
                    verbose = True
                self.evaluate(model, d.valid_data, verbose)
                if not it%2:
                    if verbose:
                        print("Test:")
                    start_test = time.time()
                    self.evaluate(model, d.test_data, verbose)
                    if verbose:
                        print(time.time()-start_test)
        
        # print(f'model.E: {(model.E.weight)}')
        torch.save(model.state_dict(), path + d.ds_name + '_state_dict.txt')

        # print(type(model.E))

        with open(path + d.ds_name + "_entity_embedding.pkl","wb") as f:
          pickle.dump(model.E,f)
          print("embedding file saved in pkl format")
        with open(path + d.ds_name + "_entity_idxs.pkl","wb") as f:
          pickle.dump(self.entity_idxs,f)
          print("entity idxs file saved in pkl format")
        
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", nargs="?",
                        help="Which dataset to use: FB15k, FB15k-237, WN18 or WN18RR.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                        help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                        help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=30, nargs="?",
                        help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                        help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                        help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                        help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                        help="Amount of label smoothing.")
    parser.add_argument("--verbose_iter", type=int, default=490,
                        help="number of iteration after which metrics will be printed.")

    
    args = parser.parse_args()
    print(args)
    
    dataset = args.dataset
    
    data_dir = "./data/%s" % dataset
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True)
    args.cuda=False
    #print(f"type of entities:: {d.entities}")
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, 
                            hidden_dropout2=args.hidden_dropout2, label_smoothing=args.label_smoothing)
    experiment.train_and_eval(args.verbose_iter)

    
                
