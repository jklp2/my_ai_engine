#include <iostream>
#include "torch.hpp"
//demoģ��,����mlp���м伤�����sigmoid
int node::node_cnt=0;
int tensor::tensor_cnt=0;
class mynet: public module{
public:
    linear l1, l2;
    Sigmoid ac;
    mynet():l1(linear(2,2,false)),l2(linear(2,2,false)){}
    //forward
    vector<vector<tensor *>> forward(vector<vector<tensor *>> x){
        auto h1=l1.forward(x);
        auto h2=ac.forward(h1);
//        print_cnt_free(h1);
        auto output=l2.forward(h2);
//        print_cnt_free(h1);
        return output;
    }
    //��¼ģ�Ͳ���
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
    auto input = T({{0.1,0.2},{0.3,0.4},{0.7,0.2}}); //��������

    auto target = T({{2,4},{1,5},{3,2}});  //��ǩ����
    mynet net1;

    optimizer G;
    G.reg({net1.get_parameters()});
    MSELOSS loss_fn;
    cout<<tensor::tensor_cnt<<endl;
    for(int i=0;i<1000000;i++){
        auto output = net1.forward(input);  //���򴫲���ͬʱ����grad_fn
//        cout<<node::node_cnt<<endl;//��������ڴ�й¶
//        cout<<tensor::tensor_cnt<<endl;//��������ڴ�й¶

//        print_cnt_free(net1.l2.w);
        auto loss = loss_fn.forward(output,target); //����loss
//        cout<<tensor::tensor_cnt<<endl;
        backward(loss); //��loss���򴫲�,����grad
        if(i%2000==0) { //����loss
            printf("iter:%d loss:%f\n", i, loss->data);
            printf("pred:\n");
            print(output);
            printf("target:\n");
            print(target);
//            print_grad(net1.l2.w);
        }
        G.step(0.2); //��grad����parameter��data
        zerograd(loss);  //��loss��ʼ�������нڵ�grad����
        free(loss);//�ͷż���ͼ��delete���еĽڵ�(module *),tensor
//        print_cnt_free(net1.l2.w);
//        cout<<node::node_cnt<<endl;//��������ڴ�й¶
//        cout<<tensor::tensor_cnt<<endl;//��������ڴ�й¶
    }
    return 0;
}
