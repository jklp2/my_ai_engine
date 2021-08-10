//
// Created by 15010 on 2021/3/6.
//

#ifndef PYTORCH_V4_LOSS_H
#define PYTORCH_V4_LOSS_H
#include "module.hpp"

class Loss:public module{

};
class MSELOSS{
public:
    tensor* forward(vector<vector<tensor*>> x, vector<vector<tensor*>> y){
        tensor* ret=NULL;
        int m=x.size(),n=x[0].size();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(!ret)
                    ret=sq(su(x[i][j],y[i][j]));
                else
                    ret=ad(ret,sq(su(x[i][j],y[i][j])));
            }
        }
        ret = di(ret,m*n);
        loss_init(ret);
        return ret;
    }
};


class L1LOSS{
public:
    tensor* forward(vector<vector<tensor*>> x, vector<vector<tensor*>> y){
        tensor* ret=NULL;
        int m=x.size(),n=x[0].size();
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(!ret)
                    ret=Abs(su(x[i][j], y[i][j]));
                else
                    ret=ad(ret,Abs(su(x[i][j], y[i][j])));
            }
        }
        ret = di(ret,m*n);
        loss_init(ret);
        return ret;
    }
};
#endif //PYTORCH_V4_LOSS_H