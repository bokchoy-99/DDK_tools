# 分类网络重训练：
1. 网络结构搜索完成后，在results目录下生成1个或多个model_*.py文件.用户可根据日志或tensorboard中的pareto图自行选择合适的网络结构
2. 将选中的.py文件拷贝到上一级目录
3. model_*.py中的网络结构默认是Pytorch实现，用户可直接使用
4. 如果用户想要使用非Pytorch框架进行训练，可直接执行```python3 model_*.py```，产生对应onnx文件，用户将网络结构翻译成所需要的版本

# Retraining of segmentation network
1. We will get one or more model_*.py files in the results directory after searching. Users can choose the appropriate network structure according to the log or Pareto Diagram in tensorboard
2. Copy the selected .py file to the previous directory
3. The network structure in model_*.py is implemented by Pytorch by default, and users can use it directly
4. If users want to use other framework for training, they can directly execute ```python3 model_*.py```. Then corresponding onnx files will be generated, and users can transform network into the required version
