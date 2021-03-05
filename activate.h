//
// Created by 15010 on 2021/3/6.
//

#ifndef PYTORCH_V4_ACTIVATE_H
#define PYTORCH_V4_ACTIVATE_H
#include "module.h"
class activate{
public:
    virtual vector<vector<tensor *>> forward(vector<vector<tensor *>>)=0;
};
class ReLu:public activate{
public:
    vector<vector<tensor *>> ret;
    vector<vector<tensor *>> forward(vector<vector<tensor *>> x){
        int m=x.size(),n=x[0].size();
        ret=vector<vector<tensor *>> (m,vector<tensor *>(n,NULL));
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                ret[i][j]=Relu({x[i][j]});
            }
        }
        return ret;
    }
};

class Sigmoid:public activate{
public:
    vector<vector<tensor *>> ret;
    vector<vector<tensor *>> forward(vector<vector<tensor *>> x){
        int m=x.size(),n=x[0].size();
        ret=vector<vector<tensor *>> (m,vector<tensor *>(n,NULL));
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                ret[i][j]= inverse({ad({Exp({Oppo({x[i][j]})})}, 1)});
            }
        }
        return ret;
    }
};
#endif //PYTORCH_V4_ACTIVATE_H
