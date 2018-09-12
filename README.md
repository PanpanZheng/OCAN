
# OCAN: One-Class Adversarial Nets for Fraud Detection

In this paper, we develop one-class adversarial nets (OCAN) for fraud detection with only benign users as training data. 

## Running Environment

The main packages you need to install are listed as follow

```
1. python 2.7 
2. tensorflow 1.3.0
```

## DateSet

For experiments, we evaluate **OCAN** on two real-world datasets: wiki and credit-card which have been attached in folder [data/](https://github.com/PanpanZheng/OCAN/tree/master/data).

## Model Evaluation

The command line for OCAN goes as follow

```
    python oc_gan.py $1 $2
```
**where** $1 refers to different datasets with wiki 1, credit-card(encoding) 2 and credit-card(plain) 3; $2 denotes whether some metrics, such as fm_loss and f1 in training process, are provided or not, with non-display 0 and display 1.


```
   e.g. python oc_gan.py 1 0 
```
The above command line shows the performance of OCAN on wiki without displaying metrics in the training process.

## Authors

* **Panpan Zheng** 

    - [personal website](https://sites.uark.edu/pzheng/)
    - [google scholar](https://scholar.google.com/citations?user=f2OLKMYAAAAJ&hl=en)

## Reference

I am very glad that you could visit this github and check my research work. If it benefits for your work, please refer this work by
.

## Acknowledgments

This work was going on underlying the guide of prof. [Xintao Wu](http://csce.uark.edu/~xintaowu/) and Dr. [Shuhan Yuan](https://sites.uark.edu/sy005/). 

Appreciate it greatly for every labmate in [**SAIL lab**](https://sail.uark.edu/)
