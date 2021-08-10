//
// Created by 15010 on 2021/3/5.
//

#ifndef PYTORCH_V4_OPTIMIZER_H
#define PYTORCH_V4_OPTIMIZER_H




//最基本的sgdoptimizer
class optimizer{
public:
    vector<tensor *> parameters;
    virtual void reg(vector<vector<tensor *>> x){
        for(auto &v:x)
            for(auto xx:v)
                parameters.push_back(xx);
    }
    virtual void step(double ratio){
        for(auto p:parameters)
            p->data-=ratio*p->grad;
    }
};
#endif //PYTORCH_V4_OPTIMIZER_H