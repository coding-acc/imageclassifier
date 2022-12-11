import os
import glob
import skimage
from skimage.feature import greycomatrix, greycoprops
import cv2
import glob
import numpy as np
from sklearn.svm import SVC
import math
from math import sqrt





def write_txt(path, txtname):

    d=[1,2]
    a=[0, np.pi/4, np.pi/2, 3*np.pi/4]


    data_path = path 
    
    txtFile = open(txtname+".txt", "a+")
    count = 0

    
        
        
    for file in glob.glob(data_path+"/*/*.jpeg"):
                          
        count+=1

        #print(file)
        if file.endswith(".jpeg") or file.endswith(".png"):
            
            
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            dim = (500, 500)
            new_img = cv2.resize(img, dim, cv2.INTER_AREA)
            lvl = new_img.max()+1
            glcm1 = np.array(greycomatrix(new_img, distances=d, angles=a, levels = lvl, normed=True, symmetric = True))
            h_features = getFeatures(glcm1)
        
            
            flat = "".join(map(lambda s: s.rstrip('\n'), str(h_features)))
            txtFile.write(flat.strip('[])')+'\n')
            
            continue
        else:
            continue
        
            
    txtFile.close

    output = "data written to text file!!"
    
    return output

def getFeatures(glcm):
    
    energy = greycoprops(glcm, 'energy')
    homogeneity = greycoprops(glcm, 'homogeneity')
    contrast = greycoprops(glcm, 'contrast')
    dissimilarity = greycoprops(glcm, 'dissimilarity')
    entropy = skimage.measure.shannon_entropy(glcm)
    feature_lista = np.append(energy.flatten(), homogeneity.flatten())
        
    feature_listb = np.append(contrast.flatten(),dissimilarity.flatten())
    feature_listc = np.append(feature_listb, dissimilarity.flatten())
    feature_list += max_p.flatten()
    feature_list = np.append(feature_lista, feature_listc)
    return feature_list


def read (txt_name):
    file = open(txt_name+".txt", "r")
    line=[]
   
        
    line = np.loadtxt(file)

    return line

class NBayes():

    def split(self, x):

        self.x_data=x
        


        self.x_normal = self.x_data[0:1340][:]
        self.x_pn = self.x_data[1340:5215][:]
        self.y_n = np.ones((1341, 1))
        self.y_p = np.zeros((3875, 1))
        
    def stats(self, x1, x2):
        
        
        m, n = x1.shape
        a, b = x2.shape

        for i in range(0,n-1):
    
            self.mean_n = np.mean(x1, axis =0)

        for j in range(0, b-1):

            self.mean_p = np.mean(x2, axis =0)

        self.mean_n = self.mean_n.reshape(1,n)
        self.mean_p = self.mean_p.reshape(1,n)
    

    def std_dev(self, x1, x2):

        m, n = x1.shape
        a, b = x2.shape

        for i in range(0,n-1):

            self.std_dev_n = np.std(x1, axis = 0)

        for j in range(0, b-1):
            
            self.std_dev_pn = np.std(x2, axis = 0)

        self.std_dev_n = self.std_dev_n.reshape(1,n)
        self.std_dev_pn = self.std_dev_pn.reshape(1,n)
        

        
    def getPDF(self, x, mean, std_dev):
        
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * std_dev))
        denominator = np.sqrt(2 * np.pi * std_dev)
        pdf = numerator/denominator

        
        return pdf

    def fit(self, x):

        self.split(x)
        
        
        
        self.stats(self.x_normal, self.x_pn)
        self.std_dev(self.x_normal, self.x_pn)
        print("fit successful - Trained")
       
        
        

    def class_prob(self, x):
        mean_n = self.mean_n
        mean_p = self.mean_p
        std_n = self.std_dev_n
        std_pn = self.std_dev_pn
        

        pdf_n = self.getPDF(x, mean_n, std_n)
        pdf_pn = self.getPDF(x, mean_p, std_pn)

        return pdf_n, pdf_pn


    def predict(self, x):
        
        pdf_norm, pdf_p = self.class_prob(x)
        m, n = x.shape
        prior_n = np.log(self.x_normal.shape[0]/5215)
        prior_p = np.log(self.x_pn.shape[0]/5215)
        
        self.post_n = []
        self.post_p = []
        self.outcome = np.zeros((m,1))

        count_n = 0
        count_p = 0
        for i in range(0, m-1):
            cond_n = np.sum(np.log(pdf_norm[i][:]))
            cond_p = np.sum(np.log(pdf_p[i][:]))
            self.post_n.append(prior_n+ cond_n)
            self.post_p.append(prior_p+ cond_p)
            y1 = np.argmax(self.post_n)
            y2 = np.argmax(self.post_p)
            if y1>y2:
                
            
                self.outcome[i] = 1
            else:

                self.outcome[i]=0



class svm_new:

    def __init__ (self, learn_rate, no_iterations=1000, lambda_param=0.5):
        self.learn_r = learn_rate
        self.iter = no_iterations
        self.lambda_p = lambda_param

    def fit(self, features):

        self.m, self.n = features.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.x = features
        self.y = np.append(np.ones((1341, 1)), np.zeros((3875, 1))).reshape(self.m, 1)

        for i in range(self.iter):
            self.update_weight()


    def update_weight(self):

        y_label = np.where(self.y <=0, -1, 1)

        for index, xi in enumerate(self.x):
            condition = y_label[index]*(np.dot(xi, self.w) - self.b) >= 1
            if (condition == True):
                dw = 2*self.lambda_p*self.w
                db = 0

            else:

                dw = 2*self.lambda_p*self.w - np.dot(xi.reshape(40,1), y_label[index])
                db = y_label[index]

            self.w = self.w - self.learn_r * dw

            self.b = self.b - self.learn_r * db


    def predict(self, x):

        output = np.dot(x, self.w)-self.b

        predic_labels = np.sign(output)

        y_ = np.where(predic_labels <=-1, 0, 1)

        return y_

class MLP:

    def __init__(self):

        np.random.seed(1)
        self.weights = 2*np.random.random((40, 1))-1
        self.y = np.append(np.ones((1341, 1)), np.zeros((3875, 1))).reshape(5216, 1)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x*(1-x)

    def train(self, train_ip, train_op, train_iter):

        for iteration in range(train_iter):
            
            output = self.logic(train_ip)
            err = train_op - output
            adjust = np.dot(np.transpose(train_ip), err*self.sigmoid_derivative(output))
            self.weights += adjust

    def logic(self, inp):

        #inputs = np.array(inp, dtype='float64')

        output = self.sigmoid(np.dot(inp, self.weights))
        return output
if __name__ == "__main__":

    mlp = MLP()
    svm = svm_new(0.1, 1000, -4)
    nb = NBayes()
    data = read("newfile")
    y = np.append(np.ones((1341, 1)), np.zeros((3875, 1))).reshape(5216, 1)
    

    choice = input("Type MLP for multi-layer perceptron, SVM for serial vector machine or NB for naive bayes:   ")
    if choice =="MLP":
        
        mlp.train(data, y, 1000)
        print("MLP trained")
        choice_1 = input("A specify text file for data or B to use provided file: ")
        if choice_1 =="A":
            txtfile = input("Enter name of textfile: ")
            test = read(txtfile)
            output = mlp.logic(test)
            print(output)

        elif choice_1 == "B":
            test = read("test")
            
            output = mlp.logic(test)
            print("Number of normal cases = "+ 100-(234-np.sum())/234*100)
            print(output)

        else:
            print("Invalid entry")


    elif choice =="SVM":
        
        svm.fit(data)
        print("SVM Trained")
       
        choice_1 = input("A specify text file for data or B to use provided file: ")
        if choice_1 =="A":
            txtfile = input("Enter name of textfile: ")
            test = read(textfile)
            output =svm.predict(test)
            print(output)

        elif choice_1 == "B":
            test = read("test")
            test = read(textfile)
            output = svm.predict(test)
            
            print(output)
            
    elif choice =="NB":
        
       nb.fit(data)
       print("NB Trained")
       
       choice_1 = input("A specify text file for data or B to use provided file: ")
       if choice_1 =="A":
           txtfile = input("Enter name of textfile: ")
           test = read(txtfile)
           output =nb.predict(test)
           print(output)

       elif choice_1 == "B":
           test = read("test")
           
           output = nb.predict(test)
            
           print(output)
    else:
        print("invalid choice")


            

    
