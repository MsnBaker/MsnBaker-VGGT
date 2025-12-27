# MsnBaker-VGGT

<div style="display: flex; gap: 10px;">
  <img src="assets/pics/pic1.png" >
  <img src="assets/pics/pic3.png" >
</div>

## English version
[🌐 English Version](README_ENGLISH.md) 

## 1. 摘要
我将VGGT的模型结合FastVGGT，预处理稍作修改，使其支持不同尺寸图片的输入。

## 2. 快速开始
### (1) 下载/克隆仓库
首先将仓库克隆到本地：
```bash
git clone https://github.com/MsnBaker/MsnBaker-VGGT.git
```

### (2) 新建conda环境 并 安装依赖包
```bash
conda create -n vggt python=3.10
conda activate vggt
cd MsnBaker-VGGT
pip install -r requirements.txt
``` 
### (3) 下载模型权重
[click me to download](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt)

然后将模型权重文件放在项目根目录下
（当然你也可以不这么做，反正你记住权重文件的路径就行了）,
命名为model.pt （应该本来就是这个名字）

### (4) 运行代码
首先将你的所有图片放在一个名为images文件夹下
进入到项目文件目录，运行
```bash
python demo_without_mask.py --data_path /PATH_TO_YOUR/images --ckpt_path /PATH_TO_YOUR/model.pt
```
上面代码的 PATH_TO_YOUR/model.pt 替换成你保存的权重文件路径. 
/PATH_TO_YOUR/images 替换成你输入图片的路径

## 3. 致谢
感谢他们的代码：  
[VGGT](https://github.com/facebookresearch/vggt)  
[FastVGGT](https://github.com/mystorm16/FastVGGT)