# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from sklearn.externals import joblib
import finger_make_change

# sklearn method
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



if __name__=="__main__":
    
    # 从实验1数据中，抽取对应数据
    for index_output_part in range(0,10):

        folder_ouput_number = './part_all/part' + str(index_output_part) + '/'

        # parameters
        startSplit, endSplit = 1,16
        littleBatchShape1 = 1176
        littleBatchShape2 = 1188
        exp_num = 1
    
        # Load Data
        data_path = '../../../../data1/jiangxingxun/zhangxilei/FingerMotion/ex'+str(exp_num)
        split_nums=6 # ori data batch
        split_nums_list = [i-1 for i in range(1,split_nums+1)]
        train_set_list, test_set_list = finger_make_change.product_set(split_nums)
    
        for circle_index in range(startSplit, endSplit+1):
            #for split_num in [0]:
            for split_num in split_nums_list:
                train_set = train_set_list[split_num]
                test_set = test_set_list[split_num]
            
                # the sample number of No.6 person is different from others
                if circle_index ==6:
                    finger = finger_make_change.finger(data_path,train_set,test_set, circle_index, littleBatchShape1, exp_num)
                else:
                    finger = finger_make_change.finger(data_path,train_set,test_set, circle_index, littleBatchShape2, exp_num)

                train_data, train_labels = finger.train_data, finger.train_labels
                test_data,test_labels=finger.test_data, finger.test_labels
            
            
                #clf = LDA() # LDA
                clf = SVC(kernel='rbf', probability=True) # SVM
                #clf = SVC(kernel='linear', probability=True) # SVM
                #clf = KNeighborsClassifier(n_neighbors=4) # KNN
                #clf = DecisionTreeClassifier(random_state=0) # DecisionTree
                #clf = MultinomialNB()  # naive_bayes
                #clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0) # randomForest
                #clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial') # Logistic



                clf.fit(train_data, train_labels)
            
                # save model and parameters
                joblib.dump(clf, folder_ouput_number+'./train_log_SVM/model/'+str(circle_index)+'/model'+str(split_num)+'.joblib')
            
                prediction = clf.predict(test_data)
                prediction_prob = clf.predict_proba(test_data)
                prediction_acc = clf.score(test_data, test_labels)
            
            
                #print "predict classification:\n",prediction
            
                print "predict proba:\n",prediction_prob
                #print "predict log(proba):\n",clf.predict_log_proba(test_data)
            
                print "acc:",prediction_acc
                prediction_acc = np.array([prediction_acc])
                #print "project data to the max class separation:\n",clf.transform([[3,2],[-1,-1]])
            
                print "predict classification:\n",prediction
            
                
                ## save file
                ### acc
                print "part number:"+str(index_output_part)
                print "saving prediction_acc, person:"+str(circle_index)+"/16,split_num:"+str(split_num)+"/5..."
                #path1 = "./train_log_LDA/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt"
                path1 = "./train_log_SVM/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt"
                #path1 = "./train_log_KNN/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt"
                #path1 = "./train_log_DecisionTree/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt"
                #path1 = "./train_log_naive_bayes/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt"
                #path1 = "./train_log_RandomForest/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt"
                #path1 = "./train_log_Logistic/acc/"+str(circle_index)+"/acc_"+str(split_num)+".txt" 
                path1 = folder_ouput_number + path1           
                np.savetxt(path1, prediction_acc)
                print "saved!" 
            
                ### prob
                print "part number:"+str(index_output_part)
                print "saving prediction_prob, person:"+str(circle_index)+"/16,split_num:"+str(split_num)+"/5..."
                #path2 = "./train_log_LDA/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                path2 = "./train_log_SVM/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                #path2 = "./train_log_KNN/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                #path2 = "./train_log_DecisionTree/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                #path2 = "./train_log_naive_bayes/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                #path2 = "./train_log_RandomForest/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                #path2 = "./train_log_Logistic/prob/"+str(circle_index)+"/prob_"+str(split_num)+".txt"
                path2 = folder_ouput_number + path2
                np.savetxt(path2, prediction_prob)
                print "saved!"
                  
                ### prob classification
                print "part number:"+str(index_output_part)
                print "saving prediction classification, person:"+str(circle_index)+"/16,split_num:"+str(split_num)+"/5..."
                #path3 = "./train_log_LDA/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                path3 = "./train_log_SVM/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                #path3 = "./train_log_KNN/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                #path3 = "./train_log_DecisionTree/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                #path3 = "./train_log_naive_bayes/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                #path3 = "./train_log_RandomForest/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                #path3 = "./train_log_Logistic/prob_classification/"+str(circle_index)+"/prob_classification_"+str(split_num)+".txt"
                path3 = folder_ouput_number + path3
                np.savetxt(path3, prediction)
                print "saved!"


                print "-"*50

        #print "LDA done!"
        print "SVM done!"
        #print "KNN done!"
        #print "DecisionTree!"
        #print "naive_bayes done!"
        #print "RandomForest done!"
        #print "Logistic done!"

        print "part number:"+str(index_output_part)+"  done!"
        
            
    
    
    


