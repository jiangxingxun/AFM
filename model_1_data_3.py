from sklearn import svm
from sklearn.externals import joblib
import finger_make
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.switch_backend('Agg')



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


    
def file_save_path(data_type ,file_type, exp_model_num, exp_test_data_num, circle_index, model_index):
    file_save_path = './log' \
        +'/model_exp'+str(exp_model_num)+'__data_exp'+str(exp_test_data_num) \
            +'/person'+str(circle_index) \
            + '/'+ data_type +'_' +str(model_index) \
                    +'__model_exp_'+str(exp_model_num) \
                    +'__data_exp_'+str(exp_test_data_num) \
                    +'__person_'+str(circle_index) + file_type
    return file_save_path



def file_save_disp(file_type, exp_model_num, exp_test_data_num, circle_index, model_index):
    file_save_disp = file_type + '='  \
        +'  model_exp:'+str(exp_model_num) \
        +', data_exp:'+str(exp_test_data_num) \
        +', person:'+ str(circle_index) \
        +', subModel_index:'+ str(model_index)  \
        +'\n' \
        +'done!'
    return file_save_disp



def confusion_disp(exp_model_num, exp_test_data_num, normalize=True):
    classes_names = ['1','2','3','4']
    if normalize==True:
        normalize_condition = 'with normalization'
    else:
        normalize_condition = 'without normalization'
    title_name = 'Confusion matrix,' + normalize_condition + ':' \
                + ' data: exp_'+str(exp_test_data_num) \
                +  ' model: exp_'+str(exp_model_num)                
    
    return classes_names, title_name   



def model_path_fun(exp_model_num, circle_index, model_index):
    model_path = './../machineLearning'+str(exp_model_num) \
                    +'/train_log_SVM/model/'+str(circle_index) \
                    +'/model'+str(model_index)+'.joblib'
            
    return model_path

def data_path(exp_test_data_num):
    data_path = '../../../../data1/jiangxingxun/zhangxilei/FingerMotion/ex'+str(exp_test_data_num)
    return data_path



if __name__=="__main__":
    
    '''
    parameters
    '''
    
    exp_test_data_num, exp_model_num = 3, 1
    start_person, end_person = 1, 16
    start_model, end_model = 0, 5
    
    # confusion matrix normalize or note
    normalize_flag = True 
    
    data_path = data_path(exp_test_data_num)
    
    
    '''
    program:
    exp_1 model 
    
    ---predict--->
    
    exp_3 data
    '''

    for circle_index in range(start_person, end_person+1):
        for model_index in range(start_model, end_model+1):
            '''
            data input
            '''
            finger = finger_make.finger(data_path, circle_index, exp_test_data_num)
            
            print('making test data...')
            test_data, test_labels = finger.test_data, finger.test_labels
            print('done...')        
    
    
            '''
            reload model
            '''
            print('loading model...')
            model_path = model_path_fun(exp_model_num,
                                    circle_index,
                                    model_index)
            clf = joblib.load(model_path)
            
            
            
            '''
            prediction and prediction_prob
            '''
            
            print('prdicting exp_'+str(exp_test_data_num)+' data using exp_'+str(exp_model_num)+' model...')
            
            
            prediction_prob = clf.predict_proba(test_data)
            prediction_prob_path = file_save_path('prediction_prob',
                                                  '.txt',
                                                  exp_model_num, 
                                                  exp_test_data_num,
                                                  circle_index,
                                                  model_index)
            np.savetxt(prediction_prob_path, prediction_prob)
            prediction_prob_txt_disp = file_save_disp('prediction_prob_txt_disp',
                                                      exp_model_num,
                                                      exp_test_data_num,
                                                      circle_index,
                                                      model_index)
            print(prediction_prob_txt_disp)
            
            
            
            prediction = clf.predict(test_data)
            prediction_path = file_save_path('prediction',
                                             '.txt',
                                             exp_model_num,
                                             exp_test_data_num,
                                             circle_index,
                                             model_index)
            np.savetxt(prediction_path, prediction)
            prediction_txt_disp = file_save_disp('prediction_txt_disp',
                                                 exp_model_num,
                                                 exp_test_data_num,
                                                 circle_index,
                                                 model_index)
            print(prediction_txt_disp)
            
            
            
            '''
            Confusion Matrix
            '''
    
            # confusion matrix
            print('making confusion matrix...')
            cnf_matrix = confusion_matrix(test_labels, prediction) # True, Predict
            print('done!')
            np.set_printoptions(precision=2)
            
            # graph of confusion matrix
            print('plot confusion matrix graph...')
            
            # normalization
            normalize=normalize_flag
            classes_names, title_name = confusion_disp(exp_model_num, exp_test_data_num, normalize=normalize)   
     
            plt.figure()
            plot_confusion_matrix(cnf_matrix, classes=classes_names, normalize=normalize, title=title_name)
            fig_path = file_save_path('confusion_matrix',
                                      '.png',
                                      exp_model_num,
                                      exp_test_data_num,
                                      circle_index,
                                      model_index)
            plt.savefig(fig_path)
            plt.close()
            
            
            confusion_matrix_fig_disp = file_save_disp('confusion_matrix_fig_disp',
                                                       exp_model_num,
                                                       exp_test_data_num,
                                                       circle_index,
                                                       model_index)
            print(confusion_matrix_fig_disp)
            print('done!')


