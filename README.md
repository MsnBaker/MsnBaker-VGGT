# MsnBaker-VGGT

## 0. 这是一个未完成的仓库！！！
## This is an unfinished repository!!!
# TODO : add a requirements.txt

![example_pic1](./assets/pics/pic1.png)

## 1. 摘要
我将FastVGGT的模型的预处理稍作修改，使其支持不同尺寸图片的输入。

## 2. 快速开始
### (1) 下载/克隆仓库
首先将仓库克隆到本地（替换为你的 GitHub 仓库地址）：
```bash
git clone git@MsnBaker.github.com:MsnBaker/MsnBaker-VGGT.git
# or 
# git clone https://github.com/MsnBaker/MsnBaker-VGGT.git
```

### (2) 新建conda环境 并 安装依赖包
```bash
conda create -n vggt python=3.10
conda activate vggt
pip install -r requirements.txt
``` 
### (3) 下载模型权重
[click me to download](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)

然后将模型权重文件放在项目根目录下
（当然你也可以不这么做，反正你记住权重文件的路径就行了）,
命名为model.pt （应该本来就是这个名字）

### (4) 运行代码
首先进入到项目文件目录，运行
```bash
python demo_without_mask.py --data_path /root/autodl-tmp/my_img/pyramid/images --ckpt_path /PATH_TO_YOUR/model.pt
```
上面代码的PATH_TO_YOUR/model.pt替换成你保存的权重文件路径

## 3. 致谢
感谢他们的代码：  
[VGGT](https://github.com/facebookresearch/vggt)  
[FastVGGT](https://github.com/mystorm16/FastVGGT)