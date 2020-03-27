### 项目二作业提交（以下commad均在project2_my_code/文件夹下执行)
说明：只是完成了项目说明中每一步的任务，没有做模型改进过程中每一次的训练结果汇总。

### Stage 1
## 任务一: 生成 train/test.txt
A：检查gt
$python generate_dataset.py --gen_data show_gt
读取I/和II/中的图片以及label.txt，并将ground truth的bbox以及关键点显示出来。

B：expand_roi
expand_roi函数实现在了Generate_dataset类里面，在进行数据标签生成时会自动调用。

D：生成数据
$python generate_dataset.py --gen_data stage1
将会在data/下面生成train.txt和test.txt两个文件，用于之后生成dataset。

E：验证
$python generate_dataset.py --gen_data stage1_inspect
将会读取刚才生成的test.txt里面的图片，并展示。

## 任务二: 网络搭建
在networks/下面定义了network_stage1.py（原始模型）。

## 任务三: 主体程序框架的搭建
完成了my_detector_stage1.py中控制流程的整体框架，包括train，test，finetune以及predict四种phase。

## 任务四:补全 FaceLandmarksDataset()
完成了my_dataloader_stage1.py，主体在FaceLadnmarksDataset类里面。

## 任务五:补全 main 函数
$python my_dataloader_stage1.py --no_normalize
将会读取test.txt里面的gt，并将resize之后的landmarks描绘在crop&resize之后的人脸上。
--no_normalize表示不用normalize作为数据增广的方法。

## 任务六:完成训练的任务
$python my_detector_stage1.py
保存路径：trained_models/OrigNet_NoNormalize_False_sgd_finetune_False/
训练后的model：model_best.pth
训练时的log：log.txt(由于使用了normalize，val_loss最后在0.055左右)
tensorboard信息：events.out.......

## 任务七:完成 Test\Predict\Finetune 代码
1. test
$python my_detector_stage1.py --pahse Test --trained_model_path trained_models/OrigNet_NoNormalize_False_sgd_finetune_False/model_best.pth

2. finetune
$python my_detector_stage1.py --is_finetune --pahse Finetune --finetune_lr 0.0005 --fine_tune_path trained_models/OrigNet_NoNormalize_False_sgd_finetune_False/model_best.pth
保存路径：trained_models/OrigNet_NoNormalize_False_sgd_finetune_True/

3. predict
$python my_detector_stage1.py --pahse Predict --trained_model_path trained_models/OrigNet_NoNormalize_False_sgd_finetune_False/model_best.pth --img_path data/test_imgs
将会调用my_predictor.py，对data/test_imgs中的图片进行关键点检测并展示。



### Stage 2
## 任务一:关于数据
1. 不做normalize
$python my_detector_stage1.py --no_normalize --optimizer adam
保存路径：trained_models/OrigNet_NoNormalize_True_adam_finetune_False/
(由于原始网络不做normalize训练时loss爆炸了，因此这里换成了adam)

2. 在my_dataloader_stage1中实现了小角度旋转的增广（其他方式没来得及做）

## 任务二:关于训练方法
通过命令行参数optimizer来实现adam，学习率设置为了sgd的5倍；
在原始网络中加入了batch_norm（由于时间问题没有观察对比无batch_norm时的差异）

## 任务三:关于网络
$python my_detector_stage1.py --no_normalize --optimizer adam --model Resnet18
使用resnet18训练网络，网络的定义放在了networks/resnet.py中。
训练结果的保存路径：trained_models/Resnet18_NoNormalize_True_adam_finetune_False_5lr/

## 任务四:关于目标(skip)
## 任务五:关于 loss
尝试了SmoothL1Loss，由于无法和MSE的loss结果做定量对比，所以没有保存训练结果，只是尝试了一下可以训练。



### Stage 3
## 任务:完成真实场景下人脸关键点检测
训练：
$python my_detector_stage3.py --no_normalize 
  --> 保存路径: trained_models/FaceNet_NoNormalize_True_adam_finetune_False/
测试：
$python my_detector_stage3.py --no_normalize --phase Test --trained_model_path trained_models/FaceNet_NoNormalize_True_adam_finetune_False/model_best.pth
  --> 测试结果：保存在stage3_test_results.png图片中

A：为网络加入分类分支
在/networks/network_stage3.py中，定义了新的针对stage3的网络，并且引入了分类和回归landmarks两个分支

B&C：生成非人脸数据，并生成新的txt文件
$python generate_dataset.py --gen_data stage3
将会在data/下面生成train_stage3.txt和test_stage3.txt两个文件，用于之后生成dataset。
格式如下：
    文件名 | 1 | 图像框 | 关键点
    文件名 | 0 | 图像框 
（非人脸数据是从原始图片中随机剪裁，判断iou之后选择出来的，在Generate_dataset类的generate_train_test_set_stage3函数中实现）

D：对于loss的处理
在my_detector_stage3.py中，定义了分类的CrossEntropyLoss和回归的MSELoss。
在train函数的85行开始，通过是否是人脸的gt来生成mask，叠加至landmarks的loss计算中
（没有使用weighted loss，因为正负样本比例比较均衡）

E：关于监测
在每一个batch的训练时记录了正负样本数，并最终计算了正负样本各自的accuracy

