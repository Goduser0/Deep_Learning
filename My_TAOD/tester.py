import torch

def classification_tester(config, ):
    test_model_path = config.test_model_path
    model_state_dict = torch.load(test_model_path)
    
    net = net.to(device)
    
    net.load_state_dict(model_state_dict)
    