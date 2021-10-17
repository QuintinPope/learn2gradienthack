# learn2gradienthack
Uses learn2learn to train a fully connected network that implements the constant function and resists being updated by gradient descent. Adapted from: https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py


Sample command: 

python3 learn_2_grad_hack.py --output_path output/ --preface output_preface --meta_lr 0.01 --meta_momentum 0.92 --base_lr 0.01 --meta_batch_size 1 --batch_size 1024 --base_steps "70,120" --meta_steps 100000 --cuda True --seed 42 --start_of_epoch_weight 1.0 --save_steps 2000 --sizes "256,128,64,64" --loss_function l1 --base_save_steps 10 --optim sgd --min_lr 0.0 --max_lr 0.0 --l2_penalty 0.0
