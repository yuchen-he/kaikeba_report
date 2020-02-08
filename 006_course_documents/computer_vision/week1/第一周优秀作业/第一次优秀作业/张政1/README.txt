【文件描述】
image文件夹中有编程所需的所有图片。

第一周_CV课作业-checkpoint.ipynb中是对作业中调查问题的回答。

以下三个程序文件均用Pycharm完成：
record course samples是作业1，复现了1.5号的课上代码。

image data augmentation是作业5.1，主要完成图片数据的增广，使用的lenna.jpg图片。你可以通过阅读
pycharm中的console中的提示内容完成本程序的使用。

convert ID's background是课上的附加作业，能够实现对身份证的前后两面的背景换底的操作。

【开发环境】
anaconda3(64bit)+pycharm11.0.5(64bit)+Python 3.7.4(64bit)+opencv4.2(64bit)

【文件说明】
image data augmentation中使用的crop，color shift，gamma change，flip，similarity transform（rotation，scale，translation），
affine transform， perspective transform。

convert ID's background中使用了mask掩模的方式，对mask进行了先腐蚀再膨胀的操作，最后使用了中值滤波。