:: test
\
python vfl_framework_for_swapattack.py \
-d CIFAR10 --path-dataset ./data/CIFAR10 \
--k 3  --half 16 \
--swap-attack True \
--epochs 20 --attack-latest-epoch 1 \
--labeled-perclass 1 --st 0.01 --batch-size 1000 --batch-swap-size 1  \
--attack-optim False --optimal-ratio 0.5 \
--specified-swap-attack True --target-label 0 


::CIFAR10
\
python vfl_framework_for_swapattack.py \
-d CIFAR10 --path-dataset ./data/CIFAR10 \
--k 3  --half 16 \
--swap-attack True \
--epochs 150 --attack-latest-epoch 50 \
--labeled-perclass 1 --st 0.01 --batch-size 32 --batch-swap-size 2  \
--attack-optim False --optimal-ratio 0.5 \
--specified-swap-attack False --target-label 0 

\
python vfl_framework_for_swapattack.py \
-d CIFAR10 --path-dataset ./data/CIFAR10 \
--k 3  --half 16 \
--swap-attack True \
--epochs 150 --attack-latest-epoch 50 \
--labeled-perclass 1 --st 0.01 --batch-size 32 --batch-swap-size 2  \
--attack-optim True --optimal-ratio 0.5 \
--specified-swap-attack False --target-label 0 