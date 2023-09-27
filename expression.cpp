#include "expression.h"

expression::expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int *inputs,
    int num_inputs):expr_id_(expr_id), op_name_(op_name), op_type_(op_type), inputs_(inputs, inputs+num_inputs)
{
    /*printf("\t\t\tExpression: expr_id %d, op_name %s, op_type %s, inputs %d (",  expr_id, op_name, op_type, num_inputs);  
    for (int i=0; i !=num_inputs; ++i)
        printf("%d," , inputs[i]);
    printf(")\n");*/
}

void expression::add_op_param_double(const char *key, double value)
{
    //std::string real_key= key;
    //constants_.insert({real_key, value});
    constants_[key]=tensor(value);
}

void expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    constants_[key]=tensor(dim,shape,data);
}
int expression::get_id(){
    return expr_id_; 
}
std:: string expression::get_op_name(){
    return op_name_;
}
std::string expression:: get_op_type(){
    return op_type_;
}
std::vector<int> expression::get_inputs(){
    return inputs_;
}
tensor expression::get_constant(std::string key){
    return constants_[key];
}



