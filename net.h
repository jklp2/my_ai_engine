//
// Created by 15010 on 2021/3/5.
//

#ifndef PYTORCH_V4_NET_H
#define PYTORCH_V4_NET_H
#include "module.h"



//类似pytorch nn.Linear
class net{
public:
    bool isbuild;
    virtual vector<tensor*> get_parameters()=0;
};


class linear:public net{
public:
    vector<vector<tensor *>> w,ret;
    vector<tensor *> b;
    bool bias;
    bool isbuild;
    linear(int n,int m,bool tag=false)     //input_size=n,output_size=m
            :bias(tag),isbuild(false)
    {   w=vector<vector<tensor *>>(n, vector<tensor *> (m));

        for(int i=0;i<n;i++) for(int j=0;j<m;j++) w[i][j]=new tensor(0.5);
        if(bias) {
            b = vector<tensor *> (m);
            for (int i = 0; i < m; i++) b[i] = new tensor(0.5);
        }
    }

    //初始化w
    void init_w(vector<vector<double>> x){
        for(int i=0;i<x.size();i++) for(int j=0;j<x[0].size();j++) w[i][j]->data=x[i][j];
    }
    //初始化b
    void init_(vector<double> x){
        for(int i=0;i<x.size();i++)  b[i]->data=x[i];
    }

    vector<vector<tensor*>> forward(vector<vector<tensor*>> &x){ //
        int ba=x.size(),n=x[0].size(),m=w[0].size();
        if(n!=(int)w.size())
            throw "linear w shape does not match the x shape";
        ret = vector<vector<tensor*>>(ba, vector<tensor*>(m, NULL));

        //矩阵乘法
        for(int i=0;i<ba;i++){
            for(int j=0;j<m;j++){
                for(int k=0;k<n;k++){
                    if(!ret[i][j])
                        ret[i][j]=mu(w[k][j],x[i][k]);
                    else
                        ret[i][j]=ad(ret[i][j],mu(w[k][j],x[i][k]));
                }
            }
        }
        //是否加上b
        if(bias){
            for(int i=0;i<ba;i++){
                for(int j=0;j<m;j++){
                    ret[i][j]=ad(ret[i][j],b[j]);
                }
            }
        }
        return ret;
    }
    //获取参数，配合optimizer使用
    vector<tensor*> get_parameters(){
        vector<tensor*> p;
        for(auto &v:w)
            for(auto x:v)
                p.push_back(x);
        if(bias)
            for(auto x:b)
                p.push_back(x);
        return p;
    }
};
#endif //PYTORCH_V4_NET_H