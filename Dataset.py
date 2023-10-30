'''
Created on Aug 8, 2016
Processing datasets. 
@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import pickle
import scipy.sparse as sp
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path):
        '''
        Constructor
        '''
        
        full_path = "/home/bclab/himchan/DeepCF-master/"+path
        self.trainMatrix = self.load_rating_file_as_matrix(full_path + ".train.rating")
        self.testMatrix = self.load_rating_file_as_matrix(full_path + ".test.rating")
        self.testRatings = self.load_rating_file_as_list(full_path + ".test.rating")
        self.testNegatives = self.load_negative_file(full_path + ".test.negative")
        self.trainDataFrame = pd.read_csv(full_path+".train.rating", header=None,sep='\t', names=['user_id', 'product_id', 'rating'])
        # self.trainReviews = pd.read_csv(full_path+".train_review.review", header=None,sep='\t', names=['user_id', 'product_id', 'review'])
        # self.totalReviews = pd.read_csv(full_path+".total_review.review", header=None,sep='\t', names=['product_id', 'review'])
        self.user_rv_dict = self.load_review_emb_dict(full_path+".user_rv_dict.pickle")
        self.item_rv_dict = self.load_review_emb_dict(full_path+".item_rv_dict.pickle")

        
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
    
    def load_review_emb_dict(self, filename):
        with open(filename, 'rb') as fr:
            rv_dict = pickle.load(fr)
        return rv_dict
