//
// Created by 15010 on 2021/3/5.
//

#ifndef PYTORCH_V4_NODE_H
#define PYTORCH_V4_NODE_H
#include "tensor.h"
#include <cmath>



class node{
public:
    tensor* ret;  //模型的输出
    vector<tensor*> input; //模型的输入
    vector<double> grad_fn; //输出对输入的导数，在forward时计算。
    bool isbuild;  //
    node():isbuild(false){}
    virtual tensor* forward(vector<tensor*> x){return NULL;}
    virtual tensor* forward(vector<tensor*> x, double q){return NULL;}
    virtual ~node(){ret->cnt_free--;ret->hook=NULL;if(ret->cnt_free==0)delete ret;}
    virtual void backward(){  //把ret的梯度传向input
        if(!isbuild)  //如果没有forward，无法传梯度。
            throw "backward before forward!";
        for(int i=0;i<input.size();i++){
            input[i]->grad+=grad_fn[i]*ret->grad;
            input[i]->cnt--;

        }
    }
};

//module中的成员定义完成，可以定义tensor的backward
void tensor::backward(){
    if(!hook)  //如果模型不存在就返回
        return;

    hook->backward();  //调用module的backward,从而把ret的梯度传向input
    for(tensor *x:hook->input){  //递归继续backward
        if(x->cnt==0)  //如果cnt不等于0，说明该tensor还有别的梯度没传过来，所以需要等待不能继续backward，比如tensor x是module a和module b的，x只收到a的梯度，还没收到b的梯度，所以x还不能继续反传。
            x->backward();
    }
}

//释放计算图，逻辑和backward类似
void tensor::free(){
    if(!hook)
        return;
    hook->backward();
    vector<tensor*> tt=hook->input;
    delete hook;
    hook=NULL; //hook置空可以防止某一节点被delete多次。
    for(tensor *x:tt){
        x->free();
    }
}

//外部free函数
void free(tensor *a){
    a->free();
}
void free(vector<tensor *> x){
    for(auto xx:x)
        xx->free();
}
void free(vector<vector<tensor *>> x){
    for(auto &v:x)
        for(auto xx:v)
            xx->free();
}



//将计算图中所有tensor grad置0
void tensor::zerograd(){
    grad=0;
    if(!hook)
        return;
    auto module=hook;
    if(module)
        for(auto &x:module->input)
            x->zerograd();
}

//乘法
class multi:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data*input[1]->data;
        grad_fn[0]=input[1]->data;
        grad_fn[1]=input[0]->data;
        input[0]->cnt++;
        input[0]->cnt_free++;
        input[1]->cnt++;
        input[1]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};

//加法
class add:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor(0);
        for(int i=0;i<input.size();i++){
            ret->data+=input[i]->data;
            grad_fn[i]=1;
            input[i]->cnt++;
            input[i]->cnt_free++;
        }
        ret->hook=this;
        return ret;
    }
    //加上常数
    tensor* forward(vector<tensor *> x, double q){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data+q;
        grad_fn[0]=1;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};


//减法
class sub:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data-input[1]->data;
        grad_fn[0]=1;
        grad_fn[1]=-1;
        input[0]->cnt++;
        input[0]->cnt_free++;
        input[1]->cnt++;
        input[1]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};

//平方
class square:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data*input[0]->data;
        grad_fn[0]=2*input[0]->data;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};


//除法
class Div:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data/input[1]->data;
        grad_fn[0]=1;
        grad_fn[1] = -input[0]->data/input[1]->data/input[1]->data;
        input[0]->cnt++;
        input[1]->cnt++;
        input[0]->cnt_free++;
        input[1]->cnt_free++;
        ret->hook=this;
        return ret;
    }
    //除以常数
    tensor* forward(vector<tensor *> x, double q){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data/q;
        grad_fn[0]=1/q;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};

//均值
class mean:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        int n=x.size();
        grad_fn.resize(n);
        input = x;
        ret = new tensor(0);
        for(int i=0;i<input.size();i++){
            ret->data+=input[i]->data;
            grad_fn[i]=1.0/n;
            input[i]->cnt++;
            input[i]->cnt_free++;
        }
        ret->data/=n;
        ret->hook=this;
        return ret;
    }
};

//relu  test
class relu:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data>0 ? input[0]->data:0;
        grad_fn[0]=input[0]->data>0 ? 1:0;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};


//exp test
class EXP:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = exp(input[0]->data);
        grad_fn[0] = exp(input[0]->data);
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};

//相反数 test
class OPPO:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = -input[0]->data;
        grad_fn[0] = -1;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};
//1/x test
class INVERSE:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = 1/input[0]->data;
        grad_fn[0] = -1/input[0]->data/input[0]->data;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};
//绝对值
class ABS:public node{
public:
    tensor* forward(vector<tensor *> x){
        isbuild=true;
        grad_fn.resize(x.size());
        input = x;
        ret = new tensor();
        ret->data = input[0]->data>0 ? input[0]->data:-input[0]->data;
        grad_fn[0]=input[0]->data>0 ? 1:-1;
        input[0]->cnt++;
        input[0]->cnt_free++;
        ret->hook=this;
        return ret;
    }
};
//外部加法
tensor* ad(tensor *a, tensor *b){
    node *op=new add();
    return op->forward({a,b});
}

//外部加法（常数）
tensor* ad(tensor *a, double q){
    node *op = new add();
    return op->forward({a},q);
}

//外部乘法
tensor* mu(tensor *a, tensor *b){
    node *op=new multi();
    return op->forward({a,b});
}

//外部减法
tensor* su(tensor *a, tensor *b){
    node *op = new sub();
    return op->forward({a,b});
}


//外部平方
tensor* sq(tensor *a){
    node *op = new square();
    return op->forward({a});
}

//外部除法
tensor* di(tensor *a, tensor *b){
    node *op = new Div();
    return op->forward({a,b});
}

//外部除法（常数）
tensor* di(tensor *a, double q){
    node *op = new Div();
    return op->forward({a},q);
}

//外部均值
tensor* me(vector<tensor *> x){
    node *op = new mean();
    return op->forward(x);
}
//外部relu test
tensor* Relu(vector<tensor *> x){
    node *op = new relu();
    return op->forward(x);
}

//外部exp test
tensor* Exp(vector<tensor *> x){
    node *op = new EXP();
    return op->forward(x);
}


//外部oppo test
tensor* Oppo(vector<tensor *> x){
    node *op = new OPPO();
    return op->forward(x);
}


//外部inverse test
tensor* inverse(vector<tensor *> x){
    node *op = new INVERSE();
    return op->forward(x);
}

//外部Abs test
tensor* Abs(tensor *a){
    node *op = new ABS();
    return op->forward({a});
}
#endif //PYTORCH_V4_NODE_H