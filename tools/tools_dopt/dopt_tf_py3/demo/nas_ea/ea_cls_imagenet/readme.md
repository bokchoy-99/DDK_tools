# 分类网络demo中加载开源依赖
## 进入分类网络demo目录
```{r, engine='bash', count_lines}
cd $DDK_PATH/tools/tools_dopt/demo/nas_ea/ea_cls_imagenet
```
## 下载开源实现
```{r, engine='bash', count_lines}
git clone https://github.com/tensorflow/models.git
```
## 进入开源代码目录
```{r, engine='bash', count_lines}
cd models
```
## 切换到指定版本
执行如下命令，使用的目标版本为 v2.1.0：
```{r, engine='bash', count_lines}
git checkout v2.1.0
```
## 返回分类网络demo目录
```{r, engine='bash', count_lines}
cd ..
```
## 设置PYTHONPATH默认路径
```{r, engine='bash', count_lines}
export PYTHONPATH=`pwd`/models/:$PYTHONPATH
```
注意：
- 每次打开终端需要重新执行一次，或添加到“~/.bashrc”文件，并执行“source ~/.bashrc”.
- `pwd`/models 路径放置在原始 $PYTHONPATH 之前，否则安装的tensorflow版本过高，会导致兼容性错误

# 分类网络重训练：

1. 网络结构搜索完成后，生成1个或多个model_arch_result_*.py文件。用户可根据日志或tensorboard中的pareto图自行选择合适的网络结构
2. 将选中的.py文件拷贝到上一级目录
3. model_arch_result_*.py中的网络结构默认是tf.keras实现，用户可直接使用
4. 如果用户想要使用非tf.keras框架进行训练，可直接执行```python3 model_arch_result_*.py```，产生可在tensorboard查看graph的可视化文件以及pb文件，用户将网络结构翻译成所需要的版本

