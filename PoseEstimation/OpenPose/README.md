1. RUN getModels.sh from command line.
2. For Python program - you can change the mode by changing the MODE to COCO / MPI 
3. For C++ - you can change the mode by changing the #define to COCO / MPI 


# AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>



基于骨架的行为识别技术的关键：
如何设计 鲁棒和有强判别性 的特征
如何 利用时域相关性 来对行为动作的动态变化进行建模。
姿态估计
（Pose Estimation）是指检测图像和视频中的人物形象的计算机视觉技术，可以确定某人的某个身体部位出现在图像中的位置，也就是在图像和视频中对人体关节的定位问题，也可以理解为在所有关节姿势的空间中搜索特定姿势。简言之，姿态估计的任务就是重建人的关节和肢干，其难点主要在于降低模型分析算法的复杂程度，并能够适应各种多变的情况、环境（光照、遮挡等等）。
输入：单帧图像
输出：一个高维的姿态向量表示关节点的位置，而不是某个类别的类标，因此这一类方法需要学习的是一个从高维观测向量到高维姿态向量的映射。

姿态估计可分为四个子方向：

单人姿态估计（Single-Person Skeleton Estimation）

单人姿态估计，首先识别出行人，然后再行人区域位置内找出需要的关键点。常见的数据集有MPII、LSP、FLIC、LIP，每种数据集都有不同的精确度指标。其中MPII是当前单人姿态估计中最常见的benchmark，使用的是PCKh指标（可以认为预测的关键点与GT标注的关键点经过head size normalize后的距离），目前有的算法已经可以在上面达到93.9%的准确率。

多人姿态估计（Multi-Person Pose Estimation）

单人姿态估计算法往往被用来做多人姿态估计，一般有两种方式。Top-down先找到图片中所有行人，然后对每个行人做姿态估计，寻找每个人的关键点；bottom-up先寻找图片中所有的parts（关键点，比如头部、手、膝盖等），然后把这些parts组装成一个个行人。

测试集主要有COCO、CrowdPose等。

人体姿态跟踪（Video Pose Tracking）

如果把姿态估计往视频中扩展，就有了人体姿态跟踪的任务。主要是针对视频场景中的每一个行人，进行人体以及每个关键点的跟踪。这是一个综合且难度较大的工作，相比于行人跟踪来说，人体关键点在视频中的temporal motion会比较大，比如一个行走的行人，手跟脚会不停的摆动，所以跟踪难度会比跟踪人体框大。

主要的数据集是PoseTrack

3D人体姿态估计（3D skeleton Estimation）

将人体姿态往3D方向进行扩展，则是输入RGB图像，输出3D的人体关键点。

经典数据集Human3.6M

除了输出3D的关键点之外，有一些工作开始研究3D的shape，比如数据集DensePose，而且从长线来讲，这个是非常有价值的研究方向。

2D姿势估计——从RGB图像估计每个关节的2D姿势（x，y）坐标。

3D姿势估计——从RGB图像估计3D姿势（x，y，z）坐标。

行为识别可以借助姿态估计的相关研究成果来实现，比如HDM05这类姿态库就提供了每一帧视频中人的骨架信息，可以基于骨架信息判断运动类型。