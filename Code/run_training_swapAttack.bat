:: test
python vfl_framework_for_swapattack.py -d CIFAR10 --path-dataset ./data/CIFAR10 \
--k 3 --epochs 1 --b 1000 --half 16 \
--swap-attack True --attack-latest-epoch 5 --labeled-perclass 1 
::CIFAR10
python vfl_framework_for_swapattack.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --epochs 10 --attack-latest-epoch 5 --half 16