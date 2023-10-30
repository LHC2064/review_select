import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from evaluate import evaluate_model_review
from Dataset import Dataset as Ds

from time import time
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='AToy_final',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()


def get_train_instances(trainMatrix, trainDataFrame, num_negatives):
    user_input, item_input, labels = [], [], []
    item_idx_list = trainDataFrame['product_id'].unique()
    for (u, i) in trainMatrix.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.choice(item_idx_list)
            while (u, j) in trainMatrix.keys():
                j = np.random.choice(item_idx_list)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return pd.DataFrame({'u_idx':user_input,'i_idx':item_input, 'label':labels})

# Dataset 상속
class CustomDataset(Dataset): 
  def __init__(self,trainMatrix, trainDataFrame, user_review_dict,item_review_dict, num_negatives):
    self.data = get_train_instances(trainMatrix, trainDataFrame, num_negatives)
    self.u_review_emb_dict = user_review_dict
    self.i_review_emb_dict = item_review_dict
  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    u_idx = self.data['u_idx'][idx]
    i_idx = self.data['i_idx'][idx]
    u_mean_tensor = self.u_review_emb_dict[u_idx]
    i_mean_tensor = self.i_review_emb_dict[i_idx]
    label = self.data['label'][idx]

    return u_mean_tensor,i_mean_tensor,label


class CFNet_review(nn.Module):
    def __init__(self):
        super(CFNet_review, self).__init__()
        #DMF
        self.dmf_user_linear1 = nn.Linear(384, 256)
        self.dmf_user_linear2 = nn.Linear(256, 128)
        self.dmf_user_linear3 = nn.Linear(128, 64)

        self.dmf_item_linear1 = nn.Linear(384, 256)
        self.dmf_item_linear2 = nn.Linear(256, 128)
        self.dmf_item_linear3 = nn.Linear(128, 64)

        #MLP
        self.mlp_user_linear = nn.Linear(384, 256)
        self.mlp_item_linear = nn.Linear(384, 256)
        
        self.mlp_concat_linear1 = nn.Linear(512, 256)
        self.mlp_concat_linear2 = nn.Linear(256, 128)
        self.mlp_concat_linear3 = nn.Linear(128, 64)

        #fusion
        self.fushion_linear = nn.Linear(128, 1)

    def forward(self, user_input, item_input):
        #DMF
        dmf_user_mlp1 = self.dmf_user_linear1(user_input)
        dmf_user_mlp2 = F.relu(self.dmf_user_linear2(dmf_user_mlp1))
        dmf_user_latent = F.relu(self.dmf_user_linear3(dmf_user_mlp2))

        dmf_item_mlp1 = self.dmf_item_linear1(item_input)
        dmf_item_mlp2 = F.relu(self.dmf_item_linear2(dmf_item_mlp1))
        dmf_item_latent = F.relu(self.dmf_item_linear3(dmf_item_mlp2))

        dmf_predictive_vector = torch.mul(dmf_user_latent, dmf_item_latent)

        #MLP
        mlp_user_latent = self.mlp_user_linear(user_input)
        mlp_item_latent = self.mlp_item_linear(item_input)
        mlp_concat_latent = torch.cat((mlp_user_latent,mlp_item_latent),dim=1)
        out = F.relu(self.mlp_concat_linear1(mlp_concat_latent))
        out = F.relu(self.mlp_concat_linear2(out))
        
        mlp_predictive_vector = F.relu(self.mlp_concat_linear3(out))

        #fusion layer
        fusion_vector = torch.cat((dmf_predictive_vector,mlp_predictive_vector),dim=1)
        pred = torch.sigmoid(self.fushion_linear(fusion_vector))

        return pred

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.epochs
    verbose = args.verbose
            
    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("DeepCF arguments: %s " % args)
    model_out_file = 'Pretrain/%s_CFNet_%d.h5' %(args.dataset, time())

    # Loading data
    t1 = time()
    dataset = Ds(args.path + args.dataset)
    trainMatrix, trainDataFrame, testRatings, testNegatives, user_rv_dict, item_rv_dict = dataset.trainMatrix, dataset.trainDataFrame, dataset.testRatings, dataset.testNegatives, dataset.user_rv_dict, dataset.item_rv_dict
    num_users, num_items = trainDataFrame['user_id'].nunique(), trainDataFrame['product_id'].nunique()  
    print("Load data done [%.1f s]. #user=%d, #item=%d, #test=%d" 
          % (time()-t1, num_users, num_items, len(testRatings)))
    
    for rep in range(10):
    # Build model
        my_model = CFNet_review().to(device) 
        loss_fn = nn.BCELoss()

        if learner.lower() == "adagrad": 
            optimizer = optim.Adagrad(my_model.parameters(), lr=0.001, weight_decay=0.0001)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(my_model.parameters(), lr=0.001, weight_decay=0.0001) 
        elif learner.lower() == "adam":
            optimizer = optim.Adam(my_model.parameters(), lr=0.001, weight_decay=0.0001) 
        else:
            optimizer = optim.SGD(my_model.parameters(), lr=0.001, weight_decay=0.0001)      
    

            
        # Check Init performance
        t1 = time()
        (hits, ndcgs) = evaluate_model_review(my_model, testRatings, testNegatives, topK, evaluation_threads,user_rv_dict,item_rv_dict)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time()-t1)) 
        
        best_hr, best_ndcg, best_iter = 0, 0, -1

        # Train model
        my_model.train()
        epochs = 150
        for epoch in range(epochs):
            t1 = time()
            # Generate training instances
            cust_dataset = CustomDataset(trainMatrix, trainDataFrame,user_rv_dict,item_rv_dict,4)
            cust_dataloader = DataLoader(cust_dataset, batch_size=1024, shuffle=True)
            # Training_torch
            best_cost = 1
            
            for batch_idx, samples in enumerate(cust_dataloader):
                u_train, i_train, y_train = samples
                # H(x) 계산
                u_train = u_train.cuda()
                i_train = i_train.cuda()
                y_train = y_train.cuda()
                y_train = y_train.float()

                prediction = my_model(u_train, i_train).squeeze(1)
                
                # cost 계산
                cost = loss_fn(prediction, y_train)
                if best_cost > cost.item():
                    best_cost = cost.item()

                # cost로 H(x) 계산
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                    epoch+1, epochs, batch_idx+1, len(cust_dataloader),
                    cost.item()
                    ))

            t2 = time()
            # Evaluation
            if epoch > 80:
                if epoch % verbose == 0:
                    (hits, ndcgs) = evaluate_model_review(my_model, testRatings, testNegatives, topK, evaluation_threads,user_rv_dict,item_rv_dict)
                    hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), best_cost
                    print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                        % (epoch+1,  t2-t1, hr, ndcg, loss, time()-t2))
                    if hr > best_hr:
                        best_hr, best_ndcg, best_iter = hr, ndcg, epoch+1
                        # if args.out > 0:
                        #     torch.save(my_model, 'best_model.pt')

        print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
        File = open("/home/bclab/himchan/DeepCF-master/results.txt", "a")
        File.write("CFNet_Review_End. Best Iteration %d:  HR = %.4f, NDCG = %.4f \n" %(best_iter, best_hr, best_ndcg))
        File.close()