#include <torch/extension.h>
#include <iostream>
#include <vector>
std::vector<at::Tensor> ncnn_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor scale,
    torch::Tensor kernel_size,
    torch::Tensor dilation,
    torch::Tensor padding,
    torch::Tensor stride,
    torch::Tensor filters,) {
    
  const auto output_h = (input.size(2) + (2*padding[0]) - ((dilation[0]*kernel_size[0]-1)+1))/stride[0] + 1
  const auto output_w = (input.size(3) + (2*padding[1]) - ((dilation[1]*kernel_size[1]-1)+1))/stride[1] + 1
      
  auto X = torch::unfold(x,kernel_size,dilation=dilation,padding=padding,stride=stride).transpose(1,2);
  
  auto X_mean = torch::mean(X,/*dim=*/-1,/*keepdim=*/true);
      
  auto X_shift = X - X_mean;
  auto X_std = torch::std(X_shift,/*dim=*/-1,/*keepdim=*/true).expand_as(X);
  auto X_refined = X_shift / (X_std+1e-5);
  
  auto flatten = weights.view(weights.size(0),-1).transpose();
  auto output = torch::addmm(bias,X_refined,flatten).view(-1,filters,output_h,output_w);

    
    
    
  return {output,
          flatten,
          X,
          X_mean,
          X_shift,
          X_std,
          X_refined};
}


std::vector<torch::Tensor> ncnn_backward(
    torch::Tensor grad_output,
    torch::Tensor grad_weights,
    torch::Tensor grad_bias,
    torch::Tensor kernel_size,
    torch::Tensor stride,
    torch::Tensor padding,
    torch::Tensor X,
    torch::Tensor gate_weights,
    torch::Tensor weights) {
    
    
    
  auto d_output_gate = torch::tanh(new_cell) * grad_h;
  auto d_tanh_new_cell = output_gate * grad_h;
  auto d_new_cell = d_tanh(new_cell) * d_tanh_new_cell + grad_cell;

  auto d_old_cell = d_new_cell;
  auto d_candidate_cell = input_gate * d_new_cell;
  auto d_input_gate = candidate_cell * d_new_cell;

  auto gates = gate_weights.chunk(3, /*dim=*/1);
  d_input_gate *= d_sigmoid(gates[0]);
  d_output_gate *= d_sigmoid(gates[1]);
  d_candidate_cell *= d_elu(gates[2]);

  auto d_gates =
      torch::cat({d_input_gate, d_output_gate, d_candidate_cell}, /*dim=*/1);

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  const auto state_size = grad_h.size(1);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}


