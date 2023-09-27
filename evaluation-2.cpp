#include <assert.h>
#include "evaluation.h"
#include "tensor.h"

evaluation::evaluation(const std::vector<expression> &exprs):exprs_(exprs)
{}

void evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    kwargs_[key] = tensor(value); 
}

void evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    kwargs_[key]=tensor(dim,shape,data);
}

int evaluation::execute()
{
    variables_.clear();

    for (auto &expr:exprs_){
        if(expr.op_type_.compare("Input")==0){
            ops_.push_back(std::make_shared<eval_input>(expr));
        }
        else if(expr.op_type_.compare("Const")==0){
             ops_.push_back(std::make_shared<eval_const>(expr));
        }
        else if(expr.op_type_.compare("Add")==0){
             ops_.push_back(std::make_shared<eval_add>(expr));
        }
        else if(expr.op_type_.compare("Sub")==0){
             ops_.push_back(std::make_shared<eval_sub>(expr));
        }
        else if(expr.op_type_.compare("Mul")==0){
             ops_.push_back(std::make_shared<eval_mul>(expr));
        }
        else if(expr.op_type_.compare("ReLU")==0){
             ops_.push_back(std::make_shared<eval_relu>(expr));
        }
        else if(expr.op_type_.compare("Flatten")==0){
             ops_.push_back(std::make_shared<eval_flatten>(expr));
        }
        else if(expr.op_type_.compare("Input2d")==0){
             ops_.push_back(std::make_shared<eval_Input2d>(expr));
        }
        else if(expr.op_type_.compare("Linear")==0){
             ops_.push_back(std::make_shared<eval_Linear>(expr));
        }
        else if(expr.op_type_.compare("MaxPool2d")==0){
             ops_.push_back(std::make_shared<eval_Maxpool2d>(expr));
        }
        else if(expr.op_type_.compare("Conv2d")==0){
             ops_.push_back(std::make_shared<eval_Convol2d>(expr));
        }

    }
    for(auto &op:ops_){
        op->eval(variables_,kwargs_);
    }
    result_=variables_[exprs_[exprs_.size()-1].expr_id_];
    return 0;
}

tensor &evaluation::get_result()
{
    return result_;
}
