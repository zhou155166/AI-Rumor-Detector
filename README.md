仅放了我的部分：
## 第一步：软件准备
Windows下win+R输入cmd回车进入命令提示符，在官网下找小于2.0.0选1.8.2：conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts后发现pytorch版本竟然不对，是因为python版本不是低于3.8。

卸载：conda remove pytorch torchvision torchaudio和conda remove cudatoolkit再conda clean --all。

重试：conda create -n new_env python=3.8和conda activate new_env并conda init后

发现python版本是降低了，但是下载pytorch会超时。考虑使用国内的镜像源，如清华大学的镜像源：

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/

conda config --set show_channel_urls yes

重新conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge成功！
## 第二步：代码试运行
conda上代码运行：conda activate new_env

F:

cd \大学\大二下\人工智能导论（A类）\大作业\我的代码\

注意这里直接cd F:\大学\大二下\人工智能导论（A类）\大作业\代码\是不会进入文件夹位置的，还是在C盘。
 
python train_model.py

但是要记得根据头文件pip install pandas scikit-learn
 
python classify.py
 
很明显效果不行，但跑通了。

自己换了BiGRU 模型后结果一般

# 同组代码
conda activate new_env1

F:

cd \大学\大二下\人工智能导论（A类）\大作业\同组代码\

python try.py

但有一些包要装

conda install pandas

conda install scikit-learn

conda install transformers -c conda-forge

发现还是报错，pillow不正常。

conda uninstall pillow

conda install pillow
 
conda install nltk
但huggingface.co现在在国内不可用，预训练模型无法下载，于是：

set HF_ENDPOINT=https://hf-mirror.com
 
但是随之而来的是AttributeError: module 'torch' has no attribute 'frombuffer'也就是PyTorch 版本与 safetensors 库不兼容。我的PyTorch原本是按照老师的1.8（<2.0），但是这里要更新的才行。

conda remove pytorch torchvision torchaudio cudatoolkit和conda clean –all后

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

python -c "import torch; print(torch.__version__)"
 
现在python try.py

调参：
hidden_dims=[256, 128, 64]时
