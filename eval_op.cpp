#include <assert.h>
#include "eval_op.h"

using namespace std;

eval_op::~eval_op() {}
eval_op::eval_op(const expression &expr) : expr_(expr)
{
    expr_id_ = expr_.get_id();
    op_name_ = expr_.get_op_name();
    inputs_ = expr_.get_inputs();
}

// constant
eval_const::eval_const(const expression &expr) : eval_op(expr), value_(expr_.get_constant("value")) {}
void eval_const::eval(vars_type &variables, const kwargs_type &kwargs)
{
    variables[expr_id_] = value_;
}

eval_input::eval_input(const expression &expr) : eval_op(expr) {}
void eval_input::eval(vars_type &variables, const kwargs_type &kwargs)
{
    variables[expr_id_] = kwargs.find(op_name_)->second;
}

eval_add::eval_add(const expression &expr) : eval_op(expr) {}
void eval_add::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_1 = variables.at(inputs_[0]);
    tensor input_2 = variables.at(inputs_[1]);
    std::vector<double> result;
    for (int i = 0; i < input_1.get_size(); ++i)
    {
        double sum_array = input_1.get_data_array()[i] + input_2.get_data_array()[i];
        result.push_back(sum_array);
    }
    variables[expr_id_] = tensor(input_1.get_dim(), input_1.get_shape_array(), &result[0]);
}

// subraction
eval_sub::eval_sub(const expression &expr) : eval_op(expr) {}
void eval_sub::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_1 = variables.at(inputs_[0]);
    tensor input_2 = variables.at(inputs_[1]);
    std::vector<double> result;
    for (int i = 0; i < input_1.get_size(); ++i)
    {
        double sub_array = input_1.get_data_array()[i] - input_2.get_data_array()[i];
        result.push_back(sub_array);
    }
    variables[expr_id_] = tensor(input_1.get_dim(), input_1.get_shape_array(), &result[0]);
}

// multipliction
eval_mul::eval_mul(const expression &expr) : eval_op(expr) {}
void eval_mul::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_1 = variables.at(inputs_[0]);
    tensor input_2 = variables.at(inputs_[1]);
    vector<double> result;
    // Value * Tensor
    if (input_1.get_dim() == 0)
    {
        double input_x = input_1.item();
        for (int i = 0; i < input_2.get_size(); ++i)
        {
            double x = input_x * input_2.get_data_array()[i];
            result.push_back(x);
        }
        variables[expr_id_] = tensor(input_2.get_dim(), input_2.get_shape_array(), &result[0]);
    }

    // Tensor * Value
    else if (input_2.get_dim() == 0)
    {
        double input_x = input_2.item();
        for (int i = 0; i < input_1.get_size(); ++i)
        {
            double x = input_x * input_1.get_data_array()[i];
            result.push_back(x);
        }
        variables[expr_id_] = tensor(input_1.get_dim(), input_1.get_shape_array(), &result[0]);
    }

    // Matrix * Matrix
    else
    {
        // initializing
        for (size_t i = 0; i < input_1.get_shape_array()[0]; ++i)
            for (size_t j = 0; j < input_2.get_shape_array()[1]; ++j)
            {
                result.push_back(0);
            }
        // multiplying matrix Input1, Input2
        for (size_t i = 0; i < input_1.get_shape_array()[0]; ++i)
            for (size_t j = 0; j < input_2.get_shape_array()[1]; ++j)
                for (size_t k = 0; k < input_1.get_shape_array()[1]; ++k)
                {
                    result[i * input_2.get_shape_array()[1] + j] += input_1.at(i, k) * input_2.at(k, j);
                }
        size_t *result_shape = input_1.get_shape_array();
        result_shape[1] = input_2.get_shape_array()[1];
        variables[expr_id_] = tensor(2, result_shape, &result[0]);
    }
}

// Relu
eval_relu::eval_relu(const expression &expr) : eval_op(expr) {}
void eval_relu::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_1 = variables.at(inputs_[0]);
    std::vector<double> result;
    for (int i = 0; i < input_1.get_size(); ++i)
    {
        if (input_1.get_data_array()[i] < 0)
        {
            result.push_back(0);
        }
        else
        {
            result.push_back(input_1.get_data_array()[i]);
        }
        
    }
    variables[expr_id_] = tensor(input_1.get_dim(), input_1.get_shape_array(), &result[0]);
}

// Flatten
eval_flatten::eval_flatten(const expression &expr) : eval_op(expr) {}
void eval_flatten::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_1 = variables.at(inputs_[0]);
    size_t N = input_1.get_shape_array()[0];
    size_t C = input_1.get_shape_array()[1];
    size_t H = input_1.get_shape_array()[2];
    size_t W = input_1.get_shape_array()[3];
    size_t result_shape[2] = {N, C * H * W};
    variables[expr_id_] = tensor(2.0, &result_shape[0], input_1.get_data_array());
}

// // Input2d
eval_Input2d::eval_Input2d(const expression &expr) : eval_op(expr) {}
void eval_Input2d::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input2d = kwargs.at(op_name_);
    size_t N = input2d.get_shape_array()[0];
    size_t H = input2d.get_shape_array()[1];
    size_t W = input2d.get_shape_array()[2];
    size_t C = input2d.get_shape_array()[3];
    vector<double> result(N * C * H * W, 0);
    for (size_t n = 0; n < N; ++n)
        for (size_t c = 0; c < C; ++c)
            for (size_t h = 0; h < H; ++h)
                for (size_t w = 0; w < W; ++w)
                {
                    result[(n * C * H * W) + (c * H * W) + (h * W) + w] = input2d.get_data_array()[(n * H * W * C) + (h * W * C) + (w * C) + c];
                }
    size_t result_shape[4] = {N, C, H, W};
    variables[expr_id_] = tensor(4.0, &result_shape[0], &result[0]);
}

// Linear
eval_Linear::eval_Linear(const expression &expr) : eval_op(expr), weight(expr_.get_constant("weight")), bias(expr_.get_constant("bias")) {}
void eval_Linear::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor linear_input = variables.at(inputs_[0]);
    size_t N = linear_input.get_shape_array()[0];
    size_t O = weight.get_shape_array()[0];
    size_t I = weight.get_shape_array()[1];
    double result = 0;
    vector<double> result2;
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t o = 0; o < O; ++o)
        {
            for (size_t i = 0; i < I; ++i)
            {
                result += weight.at(o, i) * linear_input.at(n, i);
            }
            result2.push_back(result + bias.at(o));
            result = 0;
        }
    }
    size_t result_shape[2] = {N, O};
    variables[expr_id_] = tensor(2, result_shape, &result2[0]);

}

//maxpool2d
eval_Maxpool2d::eval_Maxpool2d(const expression &expr) : eval_op(expr), kernel_size(expr_.get_constant("kernel_size").item()) {}
void eval_Maxpool2d::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_tensor = variables.at(inputs_[0]);
    vector<size_t> input_shape = input_tensor.get_shape_array_val();
    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    vector<double> input_data = input_tensor.get_data_array_val();
    vector<double> output_data; 
    double max_from_slice;
    double check;
    for (size_t example = 0; example < N; example++) {
        for (size_t channel = 0; channel < C; channel++) {
            for (size_t row = 0; row < H; row+=kernel_size) {
                for (size_t col = 0; col < W; col+=kernel_size) {
                    max_from_slice = input_tensor.at(example, channel, row, col);
                    for (size_t slice_i = 0; slice_i < kernel_size; slice_i++) {
                        for (size_t slice_j = 0; slice_j < kernel_size; slice_j++) {
                            check = input_tensor.at(example, channel, row + slice_i, col + slice_j);
                            if (check > max_from_slice) {
                                max_from_slice = check;
                            }
                        }
                    }
                    output_data.push_back(max_from_slice); 
                }
            }
        }
    }
    size_t output_shape[4] = {N, C, H/kernel_size, W/kernel_size};
    variables[expr_id_] = tensor(4, output_shape, &output_data[0]);
 
}

//convol2d
eval_Convol2d::eval_Convol2d(const expression &expr) : eval_op(expr), weight(expr_.get_constant("weight")), bias(expr_.get_constant("bias")),
            out_channels(expr_.get_constant("out_channels").item()), 
            in_channels(expr_.get_constant("in_channels").item()),
            kernel_size(expr_.get_constant("kernel_size").item()),
            padding(expr_.get_constant("padding").item()) {}
void eval_Convol2d::eval(vars_type &variables, const kwargs_type &kwargs)
{
    tensor input_tensor = variables.at(inputs_[0]);
    vector<size_t> input_shape =
        input_tensor.get_shape_array_val();
    vector<double> input_data = input_tensor.get_data_array_val();

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];
    assert((out_channels == weight.get_shape_array_val()[0]) && (out_channels == weight.get_shape_array()[0]));
    assert((in_channels == weight.get_shape_array_val()[1]) && (in_channels == C));
    assert(padding == 0);
    double conv_sum = 0.0;
    vector<double> result;
    for (size_t example = 0; example < N; example++) {
        for (size_t oc = 0; oc < out_channels; oc++) {
            for (size_t row = 0; row < H-kernel_size+1; row++) {
                for (size_t col = 0; col < W-kernel_size+1; col++) {
                    conv_sum = 0;
                    for (size_t slice_i = 0; slice_i < kernel_size; slice_i++) {
                        for (size_t slice_j = 0; slice_j < kernel_size; slice_j++) {
                            for (size_t ic = 0; ic < in_channels; ic++) {
                                conv_sum += (input_tensor.at(example, ic, row+slice_i, col+slice_j) *
                                            weight.at(oc, ic, slice_i, slice_j));
                            }
                        }
                    }
                    result.push_back(conv_sum + bias.at(oc));
                }
            }
        }
    }
    size_t result_shape[4] = {N, out_channels, H - kernel_size + 1, W - kernel_size + 1};
    variables[expr_id_] = tensor(4, result_shape, &result[0]);
}