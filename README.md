This code can be used to reproduce the experimental results of [Accordion](https://arxiv.org/abs/2010.16248). 

# Overview

Accordion borrows inspiration from the idea of critical regimes presented by [Achille et
al.](https://arxiv.org/abs/1711.08856) to improve communication efficiency of
existing gradient compression methods. Accordion uses rate of change of gradient
norm to detect Accordion keeps low compression values in
critical regimes and high compression otherwise. 

# The basic of structure of code
* The entry point of the code is *main.py*

* All the option for running code is set using either command line arguments or
  using the config dictionaries in the beginning of the *main.py*.

* All the details like data set loading, model configuration files are present
  in the **train_network** folder.

* The code for detecting crtical regimes is implemented in **auto_scale.py**

* Implementation of powerSGD and is present in
  *powersgd_grad.py*.

# Running the code
* The code is setup to run in distributed environment only. To simulate running on single GPU one can launch more than once process on single GPU.When running on single GPU or same machine you can use local host for
  master-IPFor ex- to run cifar10 training and simulate two nodes using powerSGD as a
  reducer and Accordion use the following command -

```
python main.py --model-type CNN --auto-switch --norm-file
  "cifar10_training.log" --start-k --k-start 2 --distributed --master-ip
"tcp://127.0.0.1:9998" --num-nodes 2 --rank 0
```
Run the same command again but replace --rank 0 with --rank 1 uptil 


To reproduce for example our Cifar10, ResNet-18 example run the code with 4
  nodes using the following command.
* For getting result for PowerSGD Rank 1 run-
```python
python main.py --model-type CNN --fixed-sched --norm-file "res18_psgd_k_1.log"
  --start-k --k-start 1 --distributed --master-ip "master_ip" 
--num-nodes 4 --rank 0
```

Repeat the same command on 4 different nodes but replace ``` --rank 0 ``` with 1, 2
and 3 on each node. Similarly to get result for PowerSGD Rank 2 run- 
```
python main.py --model-type CNN --fixed-sched --norm-file "res18_psgd_k_1.log"
  --start-k --k-start 2 --distributed --master-ip "master_ip"
--num-nodes 4 --rank 0
```


To get the results for Accordion run the following command
```
python main.py --model-type CNN --auto-switch --norm-file
  "res18_psgd_accordion.log" --start-k --k-start 2 --distributed --master-ip "master_ip"
--num-nodes 4 --rank 0
```

Repeat the same command on 4 different nodes but replace --rank 0 with 1, 2
and 3 on each node.


For easy reproducibility of the experiments in the paper we provide the
following bash script.
To reproduce Table 2 you can run ```./get_table_2.sh master_ip rank``` on four
different nodes providing where master_ip is of the rank 0 node and ranks range
from 0 to 3


To run more experiments users can either add more configuration dictionaries as
present at the top of the main.py or choose to modify existing ones.

# Acknowledgement 
The code borrows a lot of structure from code for 
[PowerSGD](https://github.com/epfml/powersgd). We will like to thank the authors
of PowerSGD for providing the code.

# Cite
```
@article{agarwal2020accordion,
  title={Accordion: Adaptive Gradient Communication via Critical Learning Regime Identification},
  author={Agarwal, Saurabh and Wang, Hongyi and Lee, Kangwook and Venkataraman, Shivaram and Papailiopoulos, Dimitris},
  journal={arXiv preprint arXiv:2010.16248},
  year={2020}
}
```
