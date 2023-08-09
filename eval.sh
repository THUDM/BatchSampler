export CUDA_VISIBLE_DEVICES=7
python linear.py --dataset stl10 --batch_size 512 --epochs 100 --save_path test_stl10 --model_path  ./test_stl10/128_0.5_200_256_1000_model.pth
