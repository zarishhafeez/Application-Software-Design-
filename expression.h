#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string>
#include <map>
#include "tensor.h"


class evaluation;

class expression
{
    friend class evaluation;
private: 
    int expr_id_; 
    std::string op_name_;
    std::string op_type_;
    std::vector<int> inputs_; 
    std::map<std::string, tensor> constants_;
public:
    expression(
        int expr_id,
        const char *op_name,
        const char *op_type,
        int *inputs,
        int num_inputs);

    void add_op_param_double(
        const char *key,
        double value);

    void add_op_param_ndarray(
        const char *key,
        int dim,
        size_t shape[],
        double data[]);

    int get_id();
    std::string get_op_name(); 
    std::string get_op_type();
    std::vector<int> get_inputs();
    std::map<std::string,tensor> get_op_params_tensor(); 
    tensor get_constant(std::string key);

}; // class expression

#endif // EXPRESSION_H
