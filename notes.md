1. 一维张量shape={n}和二维张量shape={1,n}在数学上是不一样的。前者只是一串纯数字，而后者是1行n列的行向量。但是他们在内存的布局是相同的
2. Linear 就是神经网络里的全连接层：把一个长度为 in 的向量，通过一个权重矩阵变成长度为 out 的新向量。
3. RMSNorm（均方根归一化）
作用：把向量的数值大小拉回稳定范围，防止经过很多层之后数值越来越大或越来越小。
4. Softmax
作用：把一组任意实数变成概率分布，也就是每个值都大于 0，而且总和为 1。
5. Silu
作用：激活函数，提供非线性特征。如果没有非线性，不管怎么线性变换都相当于一次线性变换
Silu是从ReLu演变过来的。ReLu函数中，x如果小于0，输出就是0；而SiLu不会直接是0，而是接近0
$$
\operatorname{SiLU}(x)=x\cdot\sigma(x)=\frac{x}{1+e^{-x}}
$$
6. SwiGLU
作用：也是激活函数，LLaMA的FFN用的门控激活
$$
\operatorname{SwiGLU}(x)=\operatorname{SiLU}(gate)\odot(up)
$$
7. RoPE（Rotary Position Embedding，旋转位置编码）
作用：让attention知道每个token在句子里面的位置信息。重点是要知道token之间的相对距离
