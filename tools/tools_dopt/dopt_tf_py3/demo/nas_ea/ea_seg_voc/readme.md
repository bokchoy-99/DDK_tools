# 分割网络demo中加载开源依赖
## 进入分割网络demo目录
```{r, engine='bash', count_lines}
cd $DDK_PATH/tools/tools_dopt/demo/nas_ea/ea_seg_voc
```
## 下载开源实现，重命名为models_v113
```{r, engine='bash', count_lines}
git clone https://github.com/tensorflow/models.git
mv models models_v113
```
## 进入开源代码目录
```{r, engine='bash', count_lines}
cd models_v113
```
## 切换到指定版本，使用models 1.13版本
```{r, engine='bash', count_lines}
git checkout v1.13.0
```
## 返回分割网络demo目录
```{r, engine='bash', count_lines}
cd ..
```
## 设置PYTHONPATH默认路径
```{r, engine='bash', count_lines}
export PYTHONPATH=$PYTHONPATH:`pwd`/models/research:`pwd`/models/research/slim
```
注意：每次打开终端需要重新执行一次，或添加到```~/.bashrc```文件，并执行```source ~/.bashrc```.

## 如果TensorFlow版本为2.8
### 下载开源实现，重命名为 models_v21
```{r, engine='bash', count_lines}
git clone https://github.com/tensorflow/models.git
mv models models_v21
```
### 进入开源代码目录
```{r, engine='bash', count_lines}
cd models_v21
```
### 切换到指定版本
```{r, engine='bash', count_lines}
git checkout v2.1.0
```
### 返回models_tf2.1目录
```{r, engine='bash', count_lines}
cd ..
```
### 设置PYTHONPATH默认路径
```{r, engine='bash', count_lines}
export PYTHONPATH=$PYTHONPATH:`pwd`/models_v21:`pwd`/models_v113/research:`pwd`/models_v113/research/slim
```

## 修改开源实现
### 如果TensorFlow版本为2.8，具体修改如下:
*当前路径为“ea_seg_voc”*

**step1**：修改train_utils.py，该文件的路径为：“.\models_v113\research\deeplab\utils\train_utils.py”
1. 将第72行的```slim.one_hot_encoding```修改为```tf.one_hot```
2. 在第74行的```tf.losses.softmax_cross_entropy```前面增加```return```

**step2**：修改feature_extractor.py，该文件的路径为：“.\models_v113\research\deeplab\core\feature_extractor.py”
1. 将第20和第22行代码注释掉

**step3**：修改preprocess_utils.py，该文件的路径为：“.\models_v113\research\deeplab\core\preprocess_utils.py”
1. ```tf.random_uniform```全局替换为```tf.random.uniform```
2. ```tf.reverse_v2```全局替换为```tf.reverse```
3. ```tf.to_int32```全局替换为```tf.compat.v1.to_int32```
4. ```tf.lin_space```全局替换为```tf.linspace```
5. ```tf.random_shuffle```全局替换为```tf.random.shuffle```
6. ```tf.to_float```全局替换为```tf.compat.v1.to_float```
7. ```tf.image.resize_bilinear```全局替换为```tf.compat.v1.image.resize_bilinear```
8. ```tf.image.resize_nearest_neighbor```全局替换为```tf.compat.v1.image.resize_nearest_neighbor```
9. ```tf.name_scope```全局替换为```tf.compat.v1.name_scope```

**step4**：修改utils.py，该文件的路径为：“.\models_v113\research\deeplab\core\utils.py”
1. 将第19行注释掉

**step5**：修改train_utils.py，该文件的路径为：“.\models_v113\research\deeplab\utils\train_utils.py”
1. 将第22行注释掉
2. ```tf.image.resize_bilinear```全局替换为```tf.compat.v1.image.resize_bilinear```
3. ```tf.to_float```全局替换为```tf.compat.v1.to_float```
4. ```tf.losses.softmax_cross_entropy```全局替换为```tf.compat.v1.losses.softmax_cross_entropy```

**step6**：修改input_preprocess.py，该文件的路径为：“.\models_v113\research\deeplab\input_preprocess.py”
1. 将第18行注释掉
2. 将114-115行修改为：```mean_pixel = tf.reshape([123.15, 115.90, 103.06], [1, 1, 3])```


**step7**：在“.\models_v113\research\deeplab”目录下，进行全局替换
1. `from tensorflow.contrib.slim`的调用，全局替换为 `from tf_slim`
2. `tf.contrib.slim` 的调用，全局替换为 `tf_slim`，文件首行补充 `import tf_slim`


# 分割网络重训练：
1. 网络结构搜索完成后，生成1个或多个model_arch_result_*.py文件.用户可根据日志或tensorboard中的pareto图自行选择合适的网络结构
2. 将选中的.py文件拷贝到上一级目录
3. model_arch_result_*.py中的网络结构默认是tf.keras实现，用户可直接使用
4. 如果用户想要使用非tf.keras框架进行训练，可直接执行```python3 model_arch_result_*.py```，产生可在tensorboard查看graph的可视化文件以及pb文件，用户将网络结构翻译成所需要的版本
