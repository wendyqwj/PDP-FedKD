python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism Gaussian --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 2
python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism Laplace --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 0
python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism no_dp --gpu 2