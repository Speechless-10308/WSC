###################################
#        linear probe test        #
###################################

# python linear_prob.py --data_path /home/wangjunjie/dataset --model_path /home/wangjunjie/WSC/results/cifar10/preact_resnet18_0.9/0807_2349/model_best.pth.tar

# python linear_prob.py --data_path /home/wangjunjie/dataset --model_path /home/wangjunjie/WSC/results/cifar10/preact_resnet18_0.9/0807_2349/model_best.pth.tar --noise_ratio 0.0

# python linear_prob.py --dataset cifar100 --data_path /home/wangjunjie/dataset --model_path /home/wangjunjie/WSC/results/cifar100/preact_resnet18_0.9/0817_2133/model_best.pth.tar --num_classes 100

# python linear_prob.py --dataset cifar100 --data_path /home/wangjunjie/dataset --model_path /home/wangjunjie/WSC/results/cifar100/preact_resnet18_0.9/0817_2133/model_best.pth.tar --num_classes 100 --noise_ratio 0.0

# python linear_prob.py --data_path /home/wangjunjie/dataset --model_path /mnt/f7a57ea9-f9b0-4806-966e-7f21cbc76421/wjj/logs/my-elr-checkpoint/models/cifar10_ELR_plus_PreActResNet18_sym_90/0111_215237/checkpoint-epoch200.pth --method elr+

# python linear_prob.py --data_path /home/wangjunjie/dataset --model_path /mnt/f7a57ea9-f9b0-4806-966e-7f21cbc76421/wjj/logs/my-elr-checkpoint/models/cifar10_ELR_plus_PreActResNet18_sym_90/0111_215237/checkpoint-epoch200.pth --method elr+ --noise_ratio 0.0

# python linear_prob.py --data_path /home/wangjunjie/dataset --model_path /mnt/f7a57ea9-f9b0-4806-966e-7f21cbc76421/wjj/logs/my-elr-checkpoint/models/cifar100_ELR_plus_PreActResNet18_sym_90/0112_101439/checkpoint-epoch250.pth --dataset cifar100 --num_classes 100 --method elr+

# python linear_prob.py --data_path /home/wangjunjie/dataset --model_path /mnt/f7a57ea9-f9b0-4806-966e-7f21cbc76421/wjj/logs/my-elr-checkpoint/models/cifar100_ELR_plus_PreActResNet18_sym_90/0112_101439/checkpoint-epoch250.pth --dataset cifar100 --num_classes 100 --method elr+ --noise_ratio 0.0

###################################
#       t-SNE visualization       #
###################################

# python tsne_test.py --data_path /home/wangjunjie/dataset --model_path /home/wangjunjie/WSC/results/cifar10/preact_resnet18_0.9/0825_1655/model_best.pth.tar

# python tsne_test.py --data_path /home/wangjunjie/dataset --model_path /home/wangjunjie/WSC/results/cifar100/preact_resnet18_0.9/0817_2133/model_best.pth.tar --dataset cifar100 --palette tab20 --num_classes 100

# python tsne_test.py --data_path /home/wangjunjie/dataset --model_path /mnt/f7a57ea9-f9b0-4806-966e-7f21cbc76421/wjj/logs/my-elr-checkpoint/models/cifar10_ELR_plus_PreActResNet18_sym_90/0111_215237/checkpoint-epoch200.pth --method elr+

# python tsne_test.py --data_path /home/wangjunjie/dataset --model_path /mnt/f7a57ea9-f9b0-4806-966e-7f21cbc76421/wjj/logs/my-elr-checkpoint/models/cifar100_ELR_plus_PreActResNet18_sym_90/0112_101439/checkpoint-epoch250.pth --dataset cifar100 --palette tab20 --num_classes 100 --method elr+

