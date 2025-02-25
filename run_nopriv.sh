python main_nopriv.py --dataset cifar --model cnn --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 0 --lr 0.01
python main_nopriv.py --dataset cifar --model cnn --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 2 --frac 0.2 --lr 0.01
python main_nopriv.py --dataset cifar --model cnn --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 2 --frac 0.1 --lr 0.01

python main_nopriv.py --dataset cifar --model cnn --num_users 1000 --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 1 --lr 0.01
python main_nopriv.py --dataset cifar --model cnn --num_users 1000 --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 1 --frac 0.2 --lr 0.01
python main_nopriv.py --dataset cifar --model cnn --num_users 1000 --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 2 --frac 0.1 --lr 0.01

python main_nopriv.py --dataset cifar --model cnn --num_users 10000 --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 2 --lr 0.01
python main_nopriv.py --dataset cifar --model cnn --num_users 10000 --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 0 --frac 0.2 --lr 0.01
python main_nopriv.py --dataset cifar --model cnn --num_users 10000 --epochs 100 --dp_mechanism no_dp --local_bs 5 --local_ep 100 --gpu 0 --frac 0.1 --lr 0.01