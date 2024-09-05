
# 项目名称: 情感识别项目

## 项目简介
本项目是一个基于卷积神经网络（CNN）的情感识别系统，旨在通过分析面部图像来预测人的情感状态。项目使用了TensorFlow和Keras进行模型构建和训练，并通过OpenCV处理输入的图像数据。

## 功能特性
- 支持情感识别的图像分类任务
- 通过命令行选择不同的运行模式（训练或展示结果）
- 提供模型训练过程的准确率和损失曲线可视化
- 使用图像增强技术提升模型的泛化能力
- 支持通过早停机制避免过拟合

## 依赖库
项目使用以下主要Python库：
- numpy
- argparse
- matplotlib
- OpenCV
- TensorFlow
- Keras

使用以下命令安装依赖库：
\`\`\`bash
pip install numpy argparse matplotlib opencv-python tensorflow
\`\`\`

## 项目结构
\`\`\`
- emotions.py    # 主程序，包含模型构建、训练、评估功能
- result/        # 用于存储结果的文件夹
\`\`\`

## 使用说明
### 训练模型
要训练模型，请运行以下命令：
\`\`\`bash
python emotions.py --mode train
\`\`\`

### 显示结果
如果要展示结果，使用以下命令：
\`\`\`bash
python emotions.py --mode display
\`\`\`

## 模型评估
项目提供了绘制模型训练过程中准确率和损失函数变化的函数。训练完成后，结果会自动存储在`result`文件夹中，并展示训练过程中的性能表现图。

## 文件说明
- \`emotions.py\`: 包含所有主要功能，如模型定义、数据处理、训练及可视化功能。
- \`result/\`: 该文件夹将存储模型的输出结果和图表。

