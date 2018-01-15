#/bin/bash

# train & test for tsp 300K dataset
#nohup python main.py --task=tsp --max_data_length=20 --hidden_dim=256 --train_num=300000 --test_num=30000 --train_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp1-tsp\/2DTsp_5_19_300000.txt --train_npz=.\/data\/Exp1_2DTspExp_train.npz --test_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp1-tsp\/2DTsp_5_19_30000.txt --test_npz=.\/data\/Exp1_2DTspExp_valid.npz --exp_name=Exp1_2DTspExp --input_dim=2 1>exp1.log 2>&1 &

#nohup python main.py --task=tsp --max_data_length=20 --hidden_dim=256 --train_num=300000 --test_num=60000 --train_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp1-tsp\/2DTsp_5_19_300000.txt --train_npz=.\/data\/Exp1_2DTspExp_train.npz --test_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp1-tsp\/2DTsp_5_19_60000.txt --test_npz=.\/data\/Exp1_2DTspExp_test.npz --exp_name=Exp1_2DTspExp --input_dim=2 --is_train=False --load_path=Exp1_2DTspExp_2017-11-15_12-16-38 1>exp1_test.log 2>&1 &

# train & test for path large dataset
#nohup python main.py --task=tsp --max_data_length=30 --hidden_dim=256 --train_num=62504 --test_num=13004 --train_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp3-tsp\/path_name_train.txt --test_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp3-tsp\/path_name_valid.txt --train_npz=.\/data\/Exp3_Name_train.npz --test_npz=.\/data\/Exp3_Name_valid.npz --exp_name=Exp3_Name --input_dim=18 1>exp3.log 2>&1 &
python main.py --task=tsp --max_data_length=30 --hidden_dim=256 --train_num=62504 --test_num=13004 --train_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp3-tsp\/path_name_train.txt --test_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp3-tsp\/path_name_valid.txt --train_npz=.\/data\/Exp3_Name_train.npz --test_npz=.\/data\/Exp3_Name_valid.npz --exp_name=Exp3_Name --input_dim=18

#nohup python main.py --task=tsp --max_data_length=30 --hidden_dim=256 --train_num=62504 --test_num=13331 --train_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp3-tsp\/path_name_train.txt --test_set=\/data\/zhangbin\/workspace\/pathplan\/OurData\/exp3-tsp\/path_name_test.txt --train_npz=.\/data\/Exp3_Name_train.npz --test_npz=.\/data\/Exp3_Name_test.npz --is_train=False --exp_name=Exp3_Name --input_dim=18 --load_path=Exp3_Name_2017-11-15_12-16-38 1>exp3_test.log 2>&1 &
