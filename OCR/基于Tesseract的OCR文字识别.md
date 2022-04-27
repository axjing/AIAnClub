
# 基于Tesseract的OCR文字识别

[TOC]

Tesseract是由惠普实验室开发的专有软件。2005年，它由惠普与拉斯维加斯内华达大学合作开放源代码。在过去的几年中，基于深度学习的方法在计算机视觉的许多领域中的准确性已经大大超越了传统的机器学习技术。Tesseract已实现了基于长期短期记忆（LSTM）的识别引擎。

Tesseract是一种通用的OCR引擎，但是当我们在纯白色背景上使用通用字体的纯黑色文本时，其效果最佳。当文本近似水平且文本高度至少为20像素时，它也可以很好地工作。如果文本周围有边框，则可能会将其检测为某些随机文本。

Tesseract库附带了一个方便的命令行工具，称为tesseract。我们可以使用命令行对图像进行OCR识别，并将输出存储在文本文件中。，并且Tesseract的API可以让集成在Python代码中。

## Tesseract基本用法
1. 输入文件名：在下面的示例中，我们使用image.jpg。
1. OCR国家语言：我们基本示例中的语言设置为英语（eng）。在命令行和pytesseract上，使用-l选项指定它。
1. OCR引擎模式（oem）：Tesseract 4具有两个OCR引擎-1）旧版Tesseract引擎2）LSTM引擎。使用该--oem选项可以选择四种操作模式。
```python
  0 仅旧版引擎。
  1 仅神经网络LSTM引擎。
  2 个旧版+ LSTM引擎。
  3 默认，根据可用情况。
```
1. 页面分段模式（psm）：当您具有有关文本结构的其他信息时，PSM可能非常有用。我们将在后续教程中介绍其中一些模式。在本教程中，我们将坚持psm = 3（即PSM_AUTO）。
*注意：如果未指定PSM，则在命令行和python版本中默认为3，而在C ++ API中默认为6。如果使用命令行版本和C ++ API不能获得相同的结果，请显式设置PSM。*

## Tesseract命令行用法
```bash
tesseract image.jpg stdout -l eng --oem 1 --psm 3

tesseract image.jpg output -l eng --oem 1 --psm 3

```
## 使用pytesseract实现OCR识别


## pytesseract配置方法
```python
sudo add-apt-repository ppa:alex-p/tesseract-ocr
sudo apt-get update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
sudo pip install pytesseract
```


## 代码示例
```python

import cv2
import sys
import pytesseract

if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: python ocr_simple.py image.jpg')
    sys.exit(1)
  
  # Read image path from command line
  imPath = sys.argv[1]
  #config = ('-l eng --oem 1 --psm 3')
  config = ('-l chi_sim --oem 1 --psm 3')

  # Read image from disk
  im = cv2.imread(imPath, cv2.IMREAD_COLOR)

  # Run tesseract OCR on image
  text = pytesseract.image_to_string(im, config=config)

  # Print recognized text
  print(text)
```

## 识别结果
![OCR识别原图](https://gitee.com/axjing/AnImg/raw/master/img/20201007032304.png)


![识别结果](https://gitee.com/axjing/AnImg/raw/master/img/20201007032309.png)

## 注意
出现语言包错误问题

![](https://gitee.com/axjing/AnImg/raw/master/img/20201006123806.png)

## 训练自己的语言包
https://github.com/tesseract-ocr/

>![欢迎关注：【未来现相】微信公众号！！！](https://gitee.com/axjing/AnImg/raw/master/20210808192034.png)

>未来现相WX公众号主要以下服务：
>1. 提供专业的AI解决方案及相关产品落地；
>2. 项目互助，科技创业；
>3. 每日更新AI算法研究等精品内容；
>4. 为IT从业者提供完备的职业发展规划，解决人生不同阶段的难题；