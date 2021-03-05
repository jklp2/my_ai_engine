#include <iostream>
#include "torch.h"
//定义模型
class mynet: public net{
public:
    linear l1, l2;
    Sigmoid ac;
    mynet():l1(linear(2,2,true)),l2(linear(2,2,true)){}
    //forward
    vector<vector<tensor *>> forward(vector<vector<tensor *>> x){
        auto h1=l1.forward(x);
        auto h2=ac.forward(h1);
        auto output=l2.forward(h2);
        return output;
    }
    //记录模型参数
    vector<tensor*> get_parameters(){
        auto l1_p=l1.get_parameters(),l2_p=l2.get_parameters();
        vector<tensor*> ans;
        for(auto x:l1_p)
            ans.push_back(x);
        for(auto x:l2_p)
            ans.push_back(x);
        return ans;
    }
};

int main() {
    auto input = T({{0.1,0.2},{0.3,0.4},{0.7,0.2}}); //输入例子
    auto target = T({{2,4},{1,5},{3,2}});  //标签例子
    mynet net1;
    optimizer G;
    G.reg({net1.get_parameters()});
    MSELOSS loss_fn;
    for(int i=0;i<150000;i++){
        auto output = net1.forward(input);  //正向传播，同时记算grad_fn
        auto loss = loss_fn.forward(output,target); //计算loss
        backward(loss); //从loss反向传播,计算grad
        if(i%2000==0) { //监视loss
            printf("iter:%d loss:%f\n", i, loss->data);
            printf("pred:\n");
            print(output);
            printf("target:\n");
            print(target);
        }
        G.step(0.1); //用grad更新parameter的data
        zerograd(loss);  //从loss开始反向将所有节点grad置零
        free(loss);//释放计算图，delete所有的节点(module *),tensor
    }
    return 0;
}
