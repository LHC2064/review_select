'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)
@author: hexiangnan
'''
import torch
import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np


# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, testRatings, testNegatives, K, num_thread, u_matrix, i_matrix):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    print('=================evaluate_model')
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx, u_matrix, i_matrix)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)


def evaluate_model_review(model, testRatings, testNegatives, K, num_thread, user_rv_dict, item_rv_dict):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    print('=================evaluate_model')
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_review(idx, user_rv_dict, item_rv_dict)
        hits.append(hr)
        ndcgs.append(ndcg)      
    return (hits, ndcgs)


def eval_one_rating(idx, u_matrix, i_matrix):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    u_inputs = []
    i_inputs = []
    for i in range(len(items)):
        u_inputs.append(u_matrix[u])
        i_inputs.append(i_matrix[items[i]])
    u_inputs = torch.stack(u_inputs,0).cuda()
    i_inputs = torch.stack(i_inputs,0).cuda()
    predictions = _model(u_inputs,i_inputs).squeeze()
                                 
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].item()
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)



def eval_one_review(idx, user_rv_dict, item_rv_dict):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    u_inputs = []
    i_inputs = []

    # 유저 리뷰텐서
    u_mean_tensor = user_rv_dict[u]

    for i_idx in items:
        u_inputs.append(u_mean_tensor)
        i_mean_tensor= item_rv_dict[i_idx]
        i_inputs.append(i_mean_tensor)
    u_inputs = torch.stack(u_inputs,0).cuda()
    i_inputs = torch.stack(i_inputs,0).cuda()
    predictions = _model(u_inputs,i_inputs).squeeze()

    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].item()
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(10, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)



def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
