1、第一步：在config.ini调整你想要的参数 对于三种实现方式都是通用的
2、第二步：在终端命令行安装tensorflow 跟pytorch: pip install tensorflow torch
3、第三步：分别运行tf_fnn.py，torch_fnn.py， util文件夹里面的main.py
得到三种方法的loss图 以及acc值


libsvm使用
1. 安装libsvm:  pip install -U libsvm-official
2. 运行svm.py

"""参数
-t 核函数 : 设置核函数的类型（默认为 2）
0 -- 线性
1 -- 多项式
2 -- 径向基函数
3 -- sigmoid
4 -- 预计算内核

-c c值 默认 自己选
"""