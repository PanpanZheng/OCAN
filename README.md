
# OCAN
## Command
### python oc_gan.py $1 $2
#### $1: 1 for wiki encoding (200D); 2 for credit-card encoding (50D); 3 for credit-card plain data(30D). 
#### $2: 0 for training and testing model; 1 for showing training process by testing metrics, probabilities from discriminator for benign and vandal users, fm_loss, and f1. 
#### e.g. python oc_gan.py 1 0 
#### train the model on wiki(70%), and then evaluate it(30%). 
