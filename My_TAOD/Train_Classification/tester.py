import os
import sys
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


from classification_models import classification_net_select
from trainer import Timer

##########################################################################################################
# FUNCTION: classification_tester
########################################################################################################## 
def classification_tester(config, data_iter):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    test_model_path = config.test_model_path
    checkpoint = torch.load(test_model_path)
    
    classification_net = test_model_path.split("/")[-2].split(" ")[1]
    net = classification_net_select(classification_net, config.pretrained)
    net.cuda()
    net.load_state_dict(checkpoint["model_state_dict"])
    
    timer = Timer()
    net.eval()
    
    with torch.no_grad():
        y_list = []
        y_hat_list = []
        for i, (X, y) in enumerate(data_iter):
            timer.start()
            if isinstance(X, list):
                X = [x.cuda() for x in X]
            else:
                X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            timer.stop()
            
            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                y_hat = y_hat.argmax(dim=1)
            
            y_list += list(y.cpu().numpy())
            y_hat_list += list(y_hat.cpu().numpy()) 

        Acc_score = accuracy_score(y_list, y_hat_list)
        F1_score = f1_score(y_list, y_hat_list, average='macro')
        cm = confusion_matrix(y_list, y_hat_list)
        cr = classification_report(y_list, y_hat_list)
        print(len(y_list), timer.sum())
        infer_speed = len(y_list) / timer.sum()
        # Accuracy
        print(f"Accuracy: {Acc_score}")
        # F1-score
        print(f"F1-score: {F1_score}")
        # Inference Speed
        print(f"Inference Speed: {(infer_speed / 1000.0):.3f} samples/ms")
        # Classification Report"
        print(f"Classification Report:\n{cr}")
        # Confusion_matrix
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
        plt.title('Confusion Matrix')        
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.savefig('cm.png')
        plt.close()
   
        
    