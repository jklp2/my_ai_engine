//
// Created by 15010 on 2021/3/5.
//

#ifndef PYTORCH_V4_TENSOR_H
#define PYTORCH_V4_TENSOR_H
#include <iostream>
#include <vector>



using namespace std;
class Module;

class tensor{
public:
    double data,grad;
    //cnt,cnt_free��������¼��tensor�Ǳ��ڶ��ٸ�module����Ϊ����,����cnt���������򴫲�ʱ�жϸ�tensor���ݶ�������ɼ���,��cnt_free����������moduleʱ�ͷ��ڴ�ʱ����ָ���ظ�delete�ļ�����
    int cnt,cnt_free;
    Module *hook;
    void backward();
    void zerograd();
    void free();
    void print(){cout<<data<<endl;}
    tensor(double d=0, double g=0, int c=0): data(d), grad(g), cnt(0), cnt_free(0), hook(NULL){}
    tensor(tensor &x): data(x.data), grad(x.grad), cnt(x.cnt), cnt_free(x.cnt_free), hook(x.hook){}
    tensor(tensor &&x): data(x.data), grad(x.grad), cnt(x.cnt), cnt_free(x.cnt_free), hook(x.hook){}
};

//�ⲿbackward����
void backward(tensor *a){
    a->backward();
}
void backward(vector<tensor *> x){
    for(auto xx:x)
        xx->backward();
}
void backward(vector<vector<tensor *>> x){
    for(auto &v:x)
        for(auto xx:v)
            xx->backward();
}


//�ⲿgrad���㺯��
void zerograd(tensor *a){
    a->zerograd();
}

void zerograd(vector<tensor *> x){
    for(auto xx:x)
        xx->zerograd();
}
void zerograd(vector<vector<tensor *>> x){
    for(auto &v:x)
        for(auto xx:v)
            xx->zerograd();
}

//��tensor��Ϊloss==>>��grad��Ϊdataֵ��
void loss_init(tensor *x){
    x->grad=x->data;
}
void loss_init(vector<tensor *> x){
    for(auto xx:x)
        loss_init(xx);
}
void loss_init(vector<vector<tensor *>> x){
    for(auto &v:x)
        for(auto xx:v)
            loss_init(xx);
}

//�ⲿ��ӡ
void print(vector<vector<tensor *>> w){
    for(int i=0;i<w.size();i++){
        for(int j=0;j<w[0].size();j++){
            cout<<w[i][j]->data<<" ";
        }
        cout<<endl;
    }
//    for(int i=0;i<w.size();i++){
//        for(int j=0;j<w[0].size();j++){
//            cout<<w[i][j]->grad<<" ";
//        }
//        cout<<endl;
//    }
}


//�ⲿ��������tensorָ��
tensor* T(double a){
    return new tensor(a);
}
vector<tensor*> T(vector<double> a){
    int n=a.size();
    vector<tensor*> ret(n);
    for(int i=0;i<n;i++) ret[i] = new tensor(a[i]);
    return ret;
}

vector<vector<tensor*>> T(vector<vector<double>> a){
    int m=a.size(),n=a[0].size();
    vector<vector<tensor*>> ret(m, vector<tensor*>(n));
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) ret[i][j] = new tensor(a[i][j]);
    return ret;
}
#endif //PYTORCH_V4_TENSOR_H