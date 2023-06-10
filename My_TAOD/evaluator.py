import os
import sys
import torch
import seaborn as sns
import matplotlib.pyplot as plt

<<<<<<< HEAD
from models import classification_net_select
=======
from My_TAOD.classification_models import classification_net_select
>>>>>>> 59c893ce09ef6426b0bea5b10e15d4976f1a23fa
from trainer import Timer, Accumulator, cal_correct

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)

<<<<<<< HEAD
=======

##########################################################################################################
# FUNCTION: classification_evaluator
########################################################################################################## 
>>>>>>> 59c893ce09ef6426b0bea5b10e15d4976f1a23fa
def classification_evaluator(config, data_iter):
    if not config.eval_model_path:
        sys.exit(f"ERROR:\t({__name__}): config.eval_model_path is None")
    eval_model_path = config.eval_model_path
    model_state_dict = torch.load(eval_model_path)
    
    classification_net = eval_model_path.split("/")[-1].split(" ")[1]
    net = classification_net_select(classification_net)
    net.to(config.device)
    net.load_state_dict(model_state_dict)
    
    timer = Timer()
    metric = Accumulator(2)
    net.eval()
      
    with torch.no_grad():
        y_list = []
        y_hat_list = []
        for i, (X, y) in enumerate(data_iter):
            timer.start()
            if isinstance(X, list):
                X = [x.to(config.device) for x in X]
            else:
                X = X.to(config.device)
            y = y.to(config.device)
            y_hat = net(X)
            timer.stop()
            
            metric.add(cal_correct(y_hat, y), size(y))

            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                y_hat = argmax(y_hat, axis=1)
            
            y_list += list(y.to('cpu').numpy())
            y_hat_list += list(y_hat.to('cpu').numpy()) 

        acc_score = accuracy_score(y_list, y_hat_list)
        cm = confusion_matrix(y_list, y_hat_list)
        cr = classification_report(y_list, y_hat_list)
        eval_acc = metric[0] / metric[1]
        # Accuracy
        print(acc_score)
        # Classification_report
        print(cr)  
        # Confusion_matrix
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
        plt.title('Confusion Matrix')        
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.savefig('cm.png')
        plt.close()
<<<<<<< HEAD
           
=======

>>>>>>> 59c893ce09ef6426b0bea5b10e15d4976f1a23fa
   
        
    