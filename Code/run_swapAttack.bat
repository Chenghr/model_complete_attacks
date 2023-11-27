:: test
\
python vfl_framework_for_swapattack.py \
-d CIFAR10 --path-dataset ./data/CIFAR10 \
--k 3  --half 16 \
--swap-attack True \
--epochs 20 --attack-latest-epoch 1 \
--labeled-perclass 1 --st 0.01 --batch-size 1000 --batch-swap-size 1  \
--attack-optim False --optimal-ratio 0.5


::CIFAR10
python vfl_framework_for_swapattack.py -d CIFAR10 --path-dataset ./data/CIFAR10 --k 4 --epochs 10 --attack-latest-epoch 5 --half 16