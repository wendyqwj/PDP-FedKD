python3 -u main_LDP.py --dataset mnist --model mlp --dp_mechanism MA --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 0 --dp_sample 0.01
python3 -u main_LDP.py --dataset mnist --model mlp --dp_mechanism Gaussian --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 1
python3 -u main_LDP.py --dataset mnist --model mlp --dp_mechanism Laplace --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 0
python3 -u main_LDP.py --dataset mnist --model mlp --dp_mechanism no_dp --gpu 1

python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism MA --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 0 --dp_sample 0.01
python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism Gaussian --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 1
python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism Laplace --dp_epsilon 8 --dp_delta 1e-3 --dp_clip 4 --gpu 0
python3 -u main_LDP.py --dataset cifar --model cnn --dp_mechanism no_dp --gpu 1