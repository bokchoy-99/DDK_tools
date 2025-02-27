# 检测网络demo中加载开源依赖：
## 进入检测网络demo目录：
```{r, engine='bash', count_lines}
cd $DDK_PATH/tools/tools_dopt/demo/nas_ea/ea_det_coco
```
## 下载开源实现：
```{r, engine='bash', count_lines}
git clone https://github.com/pierluigiferrari/ssd_keras.git
git clone https://github.com/tensorflow/models.git
```
## 进入ssd_keras开源代码目录：
```{r, engine='bash', count_lines}
cd ssd_keras
```
## 切换ssd_keras到指定版本：
```{r, engine='bash', count_lines}
git checkout -b v0.9.0
```
## 返回检测网络demo目录
```{r, engine='bash', count_lines}
cd ..
```
## 进入models开源代码目录
```{r, engine='bash', count_lines}
cd models
```
## 切换models到指定版本
如果TensorFlow版本为2.8，则执行如下命令，使用v2.1.0的开源版本实现：
```{r, engine='bash', count_lines}
git checkout v2.1.0
```
## 返回检测网络demo目录
```{r, engine='bash', count_lines}
cd ..
```
## 设置PYTHONPATH默认路径
```{r, engine='bash', count_lines}
export PYTHONPATH=$PYTHONPATH:`pwd`/models/
```
注意：每次打开终端需要重新执行一次，或添加到“~/.bashrc”文件，并执行“source ~/.bashrc”.


## 修改开源实现：
**step1**：修改object_detection_2d_data_generator.py。该文件的路径为：ssd_keras\data_generator\object_detection_2d_data_generator.py。
1. 第837行，```degenerate_box_handling='remove'):``` 修改为 ```degenerate_box_handling='remove', is_get_proxy=False):```
2. 第954行， ```if not (self.labels is None):``` 修改为 ```if not (self.labels is None) and (not is_get_proxy):```
3. 第1068行，```if not (self.labels is None):``` 修改为 ```if not (self.labels is None) and (not is_get_proxy):```
4. 第1095行，```if not (self.labels is None):``` 修改为 ```if not (self.labels is None) and (not is_get_proxy):```
5. 第1124行，```if not (self.labels is None): batch_y.pop(j)``` 修改为 ```if not (self.labels is None) and (not is_get_proxy): batch_y.pop(j)```
6. 第1128行，```if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)```
修改为 ```if 'original_labels' in returns and not (self.labels is None) and (not is_get_proxy): batch_original_labels.pop(j)```
7. 第31行下面，添加```sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))```

**step2**：修改keras_layer_AnchorBoxes.py。该文件的路径为：ssd_keras\keras_layers\keras_layer_AnchorBoxes.py。
1. 第21-23行，
```python
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
```
修改为：
```python
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
import tensorflow as tf
```
2 第171-174行，
```python
if K.image_dim_ordering() == 'tf':
    batch_size, feature_map_height, feature_map_width, feature_map_channels = x._keras_shape
else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
    batch_size, feature_map_channels, feature_map_height, feature_map_width = x._keras_shape
```
修改为：
```python
if K.image_data_format() == 'channels_last':
    batch_size, feature_map_height, feature_map_width, feature_map_channels = x._shape_tuple()
else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
    batch_size, feature_map_channels, feature_map_height, feature_map_width = x._shape_tuple()
```
3 第258行，```if K.image_dim_ordering() == 'tf':``` 修改为 ```if K.image_data_format() == 'channels_last':```

4 第253行，```boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))```
修改为：
```python
boxes_tensor_temp = boxes_tensor + np.zeros_like(boxes_tensor)
for _ in range(x._shape_tuple()[0] - 1):
    boxes_tensor = np.concatenate((boxes_tensor, boxes_tensor_temp), axis=0)
boxes_tensor = K.constant(boxes_tensor, dtype='float32')
```

**step3**：修改keras_layer_L2Normalization.py，该文件的路径为：ssd_keras\keras_layers\keras_layer_L2Normalization.py
1. 第47行， ```if K.image_dim_ordering() == 'tf':``` 修改为 ```if K.image_data_format() == 'channels_last':```
2. 第58行， ```self.trainable_weights = [self.gamma]``` 修改为 ```self._trainable_weights = [self.gamma]```
3. 第21-23行，
```python
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
```
修改为：
```python
import tensorflow.keras.backend as K
from tensorflow.python.keras.layers import Layer, InputSpec
```

**step4**：修改keras_ssd_loss.py，该文件的路径为：ssd_keras\keras_loss_function\keras_ssd_loss.py
1. ```tf.log```全局替换为```tf.math.log```；
2. ```tf.to_float```全局替换为```tf.compat.v1.to_float```；
3. ```tf.count_nonzero```全局替换为```tf.math.count_nonzero```；
4. ```tf.to_int32```文件替换为```tf.compat.v1.to_int32```；


# 检测网络重训练：
1. 网络结构搜索完成后，生成1个或多个model_arch_result_*.py文件。用户可根据日志或tensorboard中的pareto图自行选择合适的网络结构。
2. 将选中的.py文件拷贝到上一级目录。
3. model_arch_result_*.py中的网络结构默认是tf.keras实现，用户可直接使用。
4. 如果用户想要使用非tf.keras框架进行训练，可直接执行```python3 model_arch_result_*.py```，产生可在tensorboard查看graph的可视化文件以及pb文件，用户将网络结构翻译成所需要的版本。\r
