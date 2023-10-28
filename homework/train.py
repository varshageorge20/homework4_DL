import torch
import numpy as np

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
import inspect

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = Detector().to(device)

change = dense_transforms.Compose([
    dense_transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    dense_transforms.RandomHorizontalFlip(0.5),
    dense_transforms.ToTensor(),
    dense_transforms.ToHeatmap()
])

def train(args):
    print("hello")
    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    global_step = 0 #keeps track of the # of training steps
    torch.save(model.state_dict(), (path.join(path.dirname(path.abspath(__file__)), 'det.th')))
    
    if args.rate_ln:
        #loads the model's parameters from saved checkpoint file called det.th.
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
        # model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
        #evaluating expression; constructs dictionary which maps class names of dense_tranforms.py to corresponding class objects
        #change = eval(args.change, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})   
        
        #loading data that may undergo transformations
        data = load_detection_data('dense_data/train', transform=change)

        #MSE loss will be used during training. reduction=none means it won't be reduced to a scalar value
        mse_loss = torch.nn.MSELoss(reduction='none')

        #binary cross-entropy (BCE) loss function with logits
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        #optimizer=Adam, The model's parameters are updated using this optimizer during training
        optimizer = torch.optim.Adam(model.parameters(), lr=args.rate_gain, weight_decay=1e-4)     
        
        for epoch in range(args.count):
            model.train()
            print("i'm training")
            
            #img = image, ground truth detection maps, other=other data. these batches are provided by data loader. data=training data
            for img, ground_truth, other in data:
                
                #every 100 iterations it'll logs training progress
                if train_logger is not None and global_step % 100 == 0:
                    log(train_logger, img, ground_truth, det, global_step)

                #'loss' is the total loss
                if train_logger is not None:
                    train_logger.add_scalar('mse_loss', loss_1, global_step)
                    train_logger.add_scalar('bce_loss', loss_2, global_step)
                    train_logger.add_scalar('loss', total_loss, global_step)
                
                #calling to device to connect with gpu at runtime
                img, ground_truth, other = img.to(device), ground_truth.to(device), other.to(device)
                
                
                #This line computes the maximum values along dimension 1 (assuming that ground_truth is 
                # a tensor with multiple rows and columns) and assigns the result to the size_u variable. 
                # The keepdim=True argument keeps the dimensions in the result, ensuring that it has the 
                # same number of dimensions as the original tensor.
                size_u, _ = ground_truth.max(dim=1, keepdim=True)

                #This line appears to be running a model (model) on the img tensor and assigning the resulting 
                # values to det and size_pred
                det, size_pred = model(img)

                #It computes the mean squared error (MSE) loss between size_pred and other, multiplies it 
                # element-wise by size_u, and then takes the mean of the result. Finally, it divides the mean 
                # by the mean of size_u
                #print(size_u)
                print(size_pred)
                #loss_1 = (size_u * mse_loss(size_pred, other)) / size_u
                loss_1 = mse_loss(det, size_pred)

                #dp = torch.sigmoid(det * (1-2*ground_truth))
                
                #loss_2 = (bce_loss(det, ground_truth)).mean(): This line calculates loss_2. It computes the 
                # binary cross-entropy (BCE) loss between det and ground_truth and takes the mean of the result.
                loss_2 = bce_loss(det, ground_truth)

                # z=(bce_loss(det, ground_truth)*dp).mean()

                # z=(bce_loss(det, ground_truth)).mean()
                # o=dp.mean()
                # loss_2 = z/ o
                

                total_loss = (loss_1 + loss_2) * args.size_weight
                #total_loss =  loss_2 * args.size_weight

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                global_step = global_step + 1
            save_model(model)

    #raise NotImplementedError('train')


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    # Put custom arguments here
        #putting all the custom arguments here with their default values
    parser.add_argument('-lr','--rate_gain', type=float, default=1e-4)
    parser.add_argument('-u', '--wt', type=float, default=0.02)
    parser.add_argument('-n', '--count', type=int, default=180)
    parser.add_argument('-ln', '--rate_ln', action='store_true')
    parser.add_argument('-sw', '--size_weight', default=0.5)

    args = parser.parse_args()
    train(args)