# paipaidai_rank1

input内是数据文件。

运行环境：
jieba
keras
tensorflow
pandas
numpy

运行说明：依次运行model内1_data_aug.py，lstm_cross_char_add_feature_augmentation.py。单模十折即可达到初赛14.30-14.40（第一成绩），复赛需要将模型取不同randomseed以及char和word输入交叉取平均。
