# 分割网络demo中加载开源依赖
## 进入分割网络demo目录
```{r, engine='bash', count_lines}
cd $DDK_PATH/tools/tools_dopt/demo/nas_ea/ea_seg_voc_pytorch
```
## 下载开源实现
```{r, engine='bash', count_lines}
git clone https://github.com/pytorch/vision.git
```
## 进入开源代码目录
```{r, engine='bash', count_lines}
cd vision
```
## 切换到指定版本
```{r, engine='bash', count_lines}
git checkout v0.12.0
```
## 返回分割网络demo目录
```{r, engine='bash', count_lines}
cd ..
```
## 复制需要使用的工程文件到当前路径
```{r, engine='bash', count_lines}
cp -r vision/references .
```

# 分割网络重训练：
1. 网络结构搜索完成后，在results目录下生成1个或多个model_*.py文件.用户可根据日志或tensorboard中的pareto图自行选择合适的网络结构
2. 将选中的.py文件拷贝到上一级目录
3. model_*.py中的网络结构默认是Pytorch实现，用户可直接使用
4. 如果用户想要使用非Pytorch框架进行训练，可直接执行```python3 model_*.py```，产生对应onnx文件，用户将网络结构翻译成所需要的版本
