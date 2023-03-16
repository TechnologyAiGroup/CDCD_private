## 组内使用  
-----
#### 文件结构  
根目录下分为cdcd和ctrl文件夹。  
+ cdcd：存放所有代码文件  
    + `my_utils.py` 
    > 读取 `./ctrl/ssl/fat-results/responses/ctrl/`中的`.resp`文件生成`./cdcd/pic/`中的图片
    + `getlabel_ma.py`
    + `modify_label.py`
    > 读取`./ctrl/ssl/diagnosis_report/`中的文件生成ma尺度下的标签，并根据设置的threshold归一化标签
    + `buildDataset.py`
    > 按9:1生成测试集和训练集，存储于`./cdcd/dataset/`
    + `DAC4.py`
    > 决策树代码
    + `main.py`  
    + `data_loader.py`
    + `utils.py`
    + `model.py`
    > DANN代码
+ ctrl：存放所有原始数据  
    + ssl
        + diagnosis_report
        + fat-results
        + tmax_fail（决策树计算特征会使用到）

------
代码详细使用请看cdcd中的readme文档。
