#!usr/bin/env bash
masterip=${1?Error: no ip given}
rank=${2?Error: no rank given}

python main_bash.py --model-type cifar100 --network resnet18 --fixed-sched --norm-file "resnet18_k_1.log" --start-k --k-start 1 --distributed --master-ip $masterip --num-nodes 4 --rank $rank


python main_bash.py --model-type cifar100 --network resnet18 --fixed-sched --norm-file "resnet18_k_2.log" --start-k --k-start 2 --distributed --master-ip $masterip --num-nodes 4 --rank $rank


python main_bash.py --model-type cifar100 --network resnet18 --auto-switch --norm-file "resnet18_k_as.log" --start-k --k-start 2 --distributed --master-ip $masterip --rank $rank --num-nodes 4


python main_bash.py --model-type cifar100 --network densenet_cifar --fixed-sched --norm-file "densenet_k_1.log" --start-k --k-start 1 --distributed --master-ip $masterip --rank $rank --num-nodes 4


python main_bash.py --model-type cifar100 --network densenet_cifar --fixed-sched --norm-file "densenet_k_2.log" --start-k --k-start 2 --distributed --master-ip $masterip --rank $rank --num-nodes 4


python main_bash.py --model-type cifar100 --network densenet_cifar --auto-switch --norm-file "densenet_k_as.log" --start-k --k-start 2 --distributed --master-ip $masterip --rank $rank --num-nodes 4


python main_bash.py --model-type cifar100 --network seresnet18 --fixed-sched --norm-file "seresnet18_k_1.log" --start-k --k-start 1 --distributed --master-ip $masterip --rank $rank --num-nodes 4


python main_bash.py --model-type cifar100 --network seresnet18 --fixed-sched --norm-file "seresnet18_k_2.log" --start-k --k-start 2 --distributed --master-ip $masterip --rank $rank --num-nodes 4


python main_bash.py --model-type cifar100 --network seresnet18 --auto-switch --norm-file "seresnet18_k_as.log" --start-k --k-start 2 --distributed --master-ip $masterip --rank $rank --num-nodes 4

echo "ResNet-18, Cifar100, K=1 results"
python parse_output.py resnet18_k_1.log
echo "ResNet-18, Cifar100, K=2 results"
python parse_output.py resnet18_k_2.log
echo "ResNet-18, Cifar100, AdaSparse Results"
python parse_output.py resnet18_k_as.log
echo "DenseNet, Cifar100, K=1 results"
python parse_output.py densenet_k_1.log
echo "DenseNet, Cifar100, K=2 results"
python parse_output.py densenet_k_2.log
echo "DenseNet, Cifar100, AdaSparse Results"
python parse_output.py densenet_k_as.log
echo "SeNet, Cifar100, K=1 results"
python parse_output.py seresnet18_k_1.log
echo "SeNet, Cifar100, K=2 results"
python parse_output.py seresnet18_k_2.log
echo "SeNet, Cifar100, AdaSparse Results"
python parse_output.py seresnet18_k_as.log
