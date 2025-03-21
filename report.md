# 项目报告

## ai chat
实现ai chat功能；将多段对话看做连续的文本内容，生成新的内容；通过kv cache存储连续的文本信息。

## 混合精度运算

首先考虑到混合精度运算会导致计算精度变低，因此一些对精度敏感的算子内部还是会用fp32计算，如：rms_norm, softmax, rope等
再者我参考了在deepseek的FP8中为了避免累加导致精度下溢的问题，在混合精度实现时求和使用fp32存储；这样虽然能提升累加精度，但是会引入大量的精度转化；
由于half crate对于精度的转化和fp16的乘法并没有硬件支持，再加上提升累加精度引入的大量精度转化，所以在实现混合精度运算后，模型的运行速度成倍的下降；

**如何解决这种问题**
- 结合硬件对于精度转化的速度以及精度的损失，找到精度与速度的合理平衡点；不适合存在过多精度转化；
- FP16与FP8相比较，精度的损失并没有那么严重，可能并没有引入提升累加精度的必要；需要寻找论文数据或实验支持；

## 分布式推理

- 对于Attention层，按照注意力头进行分布式运算，并在得出每一个注意力头的注意力分数后，与o_proj行切割后的矩阵进行矩阵乘，并在最后进行allReduce连接，并且只有守护线程进行一次残差累加；
- 对于MLP层类似，将参数进行行切割和列切割，切割的对象是intermediate_size，按照CPU的核心数量进行切割，在完成down矩阵乘积后，进行allReduce连接操作，同样只有守护线程进行一次残差累加；

对于同一输入输出：
分布式推理：
Time elapsed is: 80.92310627s
常规推理：
Time elapsed is: 127.950527394s

分布式推理过程中各个CPU负载均衡，但是都未达到100%，平均在50%；在常规的推理中可以有一个CPU核心达到100%。

![alt text](image.png)

**可能如何解决这种问题**

- 优化数据的走向，尽可能减少数据拷贝
- 负载均衡，使得每一个CPU处理量类似
- 寻找限制CPU使用率上升的原因
- 在allReduce阶段只是单个核心工作

