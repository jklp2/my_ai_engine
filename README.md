# my_ai_engine
参考pytorch和tensorflow，自己用c++搭的一个深度学习框架，自学用，下载后把工程目录改名为pytorch_v4
## 主要逻辑：  
任何函数都可以表示为计算图：  
![image](https://github.com/jklp2/my_ai_engine/blob/main/1.png)  
边是tensor，注意两条不同的边可以来自同一节点，这样代表它们是同一个tensor。  
节点是运算（node.h中定义），多输入单输出，节点维护图的连接，节点在计算时同时维护输出关于所有输入的梯度。  
节点负责维护forward和backward。  
任何多输入多输出的复杂模型都可以用这种计算图描述。  
## 实现细节
### tensor:  
>主要记录数值data;loss关于该tensor的梯度grad;和来自哪个节点hook。  
>不负责维护图的拓扑关系。（pytorch的策略，tensorflow 1的图是由tensor维护的）。 
>反向传播时从下级的节点获取节点输出的梯度ret->grad以及输出关于自身的梯度，grad_fn。两者相乘（链式法则）就是该下级节点贡献的梯度。  
>将所有下级节点贡献的梯度求和就是当前tensor的梯度。  

### node:   
>对应图中的节点,负责运算，node.h里定义。  
>节点的输出只有一个tensor(ret)，输入可以是多个tensor(input)，同时需要记录输出关于每个输入的梯度grad_fn。
>在正向运算(forward)的同时，维护输出对于各个输入的梯度grad_fn。
>反向传播时把ret的grad乘上每个输入的梯度grad_fn,就能获知每个输入收到的梯度。
>必须在ret已经收到所有梯度后才能触发反向传播。  
>这是维护forward和backward基本单位，module是若干节点的封装。  
>
### module：
>计算图的子图，可以视为多输入多输出函数，也就是pytorch中的nn.Module。
>需要有参数。

### loss：
>损失函数，特殊的module，和module不同的是输出的梯度自动设为输出的数值。 

### activate：
>激活函数，特殊的module，和module不同的是没有参数。
 
### optimizer：
>将可训练参数更新的工具

## demo
>见main.cpp,两层的全链接mlp，中间激活函数是sigmoid。

