# Tensorflow2学习笔记

这是一个Tensorflow2的学习笔记。拥有最全面的中文注释和代码示例。

## 项目概述

- 该项目是一个全面的TensorFlow 2学习笔记，旨在为初学者和进阶开发者提供从基础到高级的TensorFlow 2教程。项目中包含丰富的中文注释和代码示例，帮助用户更好地理解和掌握TensorFlow 2的核心概念和应用。
  
  主要内容

- 基础教程：涵盖TensorFlow 2的基本概念、安装配置、简单的线性回归、多层网络等基础知识。
- 深度学习模型：包括逻辑回归、softmax多分类、卷积神经网络（CNN）、循环神经网络（RNN）等多种深度学习模型的实现和讲解。
- 高级主题：涉及Eager模式与自定义训练、图像定位、目标检测、自编码器和变分编码器等内容，帮助用户深入理解TensorFlow 2的高级特性。
- 实用案例：通过实际案例如Fashion MNIST数据集分类、猫狗分类、电影评论情感分析等，展示如何将理论应用于实践。

## 目录

- [版本](#版本)
- [简单的线性回归.py](#简单的线性回归py)
- [多层网络.py](#多层网络py)
- [逻辑回归.py](#逻辑回归py)
- [softmax多分类和fashion_mnist数据集.py](#softmax多分类和fashion_mnist数据集py)
- [keras函数式api.py](#keras函数式apipy)
- [tf_data模块学习.py](#tf_data模块学习py)
- [tf_data实例-手写数字识别.py](#tf_data实例-手写数字识别py)
- [卷积神经网络CNN](#卷积神经网络cnn)
  - [data](#data)
  - [img](#img)
  - [1.学习卷积.py](#1学习卷积py)
  - [2.识别fashin_mnist.py](#2识别fashin_mnistpy)
  - [3.湖泊飞机图片分类（二分类）.py](#3湖泊飞机图片分类二分类py)
  - [4.200种鸟类图片分类（多分类）.py](#42000种鸟类图片分类多分类py)
  - [5.tf.keras序列问题-电影评论.py](#5tfkeras序列问题-电影评论py)
- [Eager模式与自定义训练](#eager模式与自定义训练)
  - [通透！必会的 6 大卷积神经网络架构_files](#通透必会的-6-大卷积神经网络架构_files)
  - [1.eager模式.py](#1eager模式py)
  - [2.变量与自动微分运算.py](#2变量与自动微分运算py)
  - [3.metrics汇总计算模块.py](#3metrics汇总计算模块py)
  - [4.TensorBoard可视化.py](#4tensorboard可视化py)
  - [5.可视化之自定义标量.py](#5可视化之自定义标量py)
  - [6.自定义训练中使用可视化.py](#6自定义训练中使用可视化py)
  - [7.自定义训练案例之猫狗分类.py](#7自定义训练案例之猫狗分类py)
  - [8.使用预训练网络（迁移学习-猫狗分类-vgg16）.py](#8使用预训练网络迁移学习-猫狗分类-vgg16py)
  - [9.多输出模型实例-颜色&衣服分类.py](#9多输出模型实例-颜色衣服分类py)
  - [10.多输出模型实例-自定义损失.py](#10多输出模型实例-自定义损失py)
  - [11.模型的保存与恢复.py](#11模型的保存与恢复py)
  - [12.模型的保存与恢复2.py](#12模型的保存与恢复2py)
  - [13.模型的保存与恢复3.py](#13模型的保存与恢复3py)
- [图像定位](#图像定位)
  - [1.图像定位-宠物头部画框.py](#1图像定位-宠物头部画框py)
  - [2.自动图运算模式（Graph Execution）.py](#2自动图运算模式graph-executionpy)
  - [3.图像语义分割-宠物轮廓.py](#3图像语义分割-宠物轮廓py)
  - [4.自定义层.py](#4自定义层py)
  - [5.自定义模型.py](#5自定义模型py)
  - [6.Unet语意分割模型-城市景物分类.py](#6unet语意分割模型-城市景物分类py)
  - [7.残差结构与Resnet.py](#7残差结构与resnetpy)
  - [8.Linknet图像语义分割模型.py](#8linknet图像语义分割模型py)
- [循环神经网络RNN](#循环神经网络rnn)
  - [1.循环神经网络.py](#1循环神经网络py)
  - [2.空气污染预测.py](#2空气污染预测py)
  - [3.一维卷积.py](#3一维卷积py)
  - [4.叶子分类-一维卷积优化.py](#4叶子分类-一维卷积优化py)
- [自编码器和变分编码器-手写数字图像生成](#自编码器和变分编码器-手写数字图像生成)
  - [1.自编码器.py](#1自编码器py)
  - [2.变分自编码器.py](#2变分自编码器py)
  - [变分编码器结构.jpg](#变分编码器结构jpg)
- [目标检测与Object Detection API](#目标检测与object-detection-api)
  - [1目标识别.py](#1目标识别py)

## 安装

使用以下命令安装依赖：

```bash
pip install numpy, pandas，matplotlib，tensorboard，tensorflow-datasets，tensorflow, glob, lxml
```

## 使用说明

- 阅读文档：每个章节都配有详细的中文注释和解释，建议按顺序阅读并理解每一部分内容。
- 运行代码：确保所有依赖已正确安装后，可以直接运行各个Python脚本文件，体验TensorFlow 2的强大功能。
- 数据集：数据集只为展示数据的形状，并不全面，如果需要可自行联系本人。
- 参与贡献：欢迎对项目提出改进建议或贡献新的内容，共同完善这个学习资源。
通过以上扩展描述，希望用户能够更全面地了解该项目的内容和价值，从而更好地利用这些资源进行学习和实践。

## 构建方法

本项目使用Python编写，无需构建过程。

## 贡献者

- [日月光华](https://study.163.com/course/courseMain.htm?courseId=1004573006&_trace_c_p_k2_=a545557bf66342f7b06c4c2de9a029d7)
- 本教程根据日月光华老师的视频教程进行学习，并做了修改。大家可以支持下老师。讲解非常全面，非常通俗！

## 许可证

本项目采用MIT许可证。请查看项目中的LICENSE文件了解更多信息。

## 注意事项

- 本项目仅供学习和研究使用，请勿用于商业用途。
- 请遵守酷我音乐网站的使用条款和条件，不要违反任何法律法规。
- 本项目不保证爬取到的数据完全准确，请自行验证。

## 贡献

如果您有任何建议或改进意见，请随时提交Pull Request或创建Issue。

## 免责声明

本项目不承担因使用本项目而导致的任何法律责任。请用户自行承担风险。

## 联系方式

如果您有任何问题或建议，请通过以下方式联系我们：

- 邮箱：[2262752545@qq.com](mailto:2262752545@qq.com)
- 网站：[https://space.bilibili.com/543167653](https://space.bilibili.com/543167653)

# Engerlish Version


## Introduction

This project is a comprehensive guide to TensorFlow 2, covering basic concepts, installation, simple linear regression, multi-layer networks, and various deep learning models. It also includes advanced topics such as Eager mode and custom training, image localization, object detection, autoencoders, and variational autoencoders. The project is designed to help users understand TensorFlow 2 and its applications in practice.

## Installation

Use the following command to install dependencies:
```bash
pip install numpy, pandas，matplotlib，tensorboard，tensorflow-datasets，tensorflow, glob, lxml
```

## Usage

- Read the documentation: Each chapter comes with detailed Chinese comments and explanations, and it is recommended to read and understand each part in order.
- Run the code: After ensuring that all dependencies are correctly installed, you can directly run the various Python script files to experience the powerful features of TensorFlow 2.
- Dataset: The dataset is only for displaying the shape of the data and is not comprehensive. If needed, you can contact the author for more information.

## Contributors

- [日月光华](https://study.163.com/course/courseMain.htm?courseId=1004573006&_trace_c_p_k2_=a545557bf66342f7b06c4c2de9a029d7)
- This tutorial is based on the video tutorials of the teacher, and some modifications have been made. Users are welcome to support the teacher. The explanation is very comprehensive and easy to understand!
  
## Contribution

If you have any suggestions or improvements, please submit a Pull Request or create an Issue.

## License

This project is licensed under the MIT License. Please see the LICENSE file for more details.
## Disclaimer

This project does not guarantee the accuracy of the data obtained through crawling, please verify yourself.
## Contact

If you have any questions or suggestions, please contact us through the following methods:
- Email: [2262752545@qq.com](mailto:2262752545@qq.com)
- Website: [https://space.bilibili.com/543167653](https://space.bilibili.com/543167653)
  