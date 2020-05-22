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
    
  auto X_scaled = scale*X_refined
  
  
  auto flatten = weights.view(weights.size(0),-1).transpose();
  auto output = torch::addmm(bias,X_scaled,flatten).view(-1,filters,output_h,output_w);

    
    
    
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
    torch::Tensor kernel_size,
    torch::Tensor stride,
    torch::Tensor padding,
    torch::Tensor X,
    torch::Tensor X_mean,
    torch::Tensor X_std,
    torch::Tensor X_refined,
    torch::Tensor X_scaled,
    torch::Tensor weights) {
    
    
  const auto numb = 1.0/kernel_size[0]*kernel_size[1]*X.size(2);
  auto grad_output_ = grad_ouput*scale
  auto grad_sigma = torch::sum(grad_output_ * (X-X_mean),/*axis=*/=-1,/*keepdim=*/=True)*0.5*(X_std)**(-1.5);
  auto grad_mean = torch::sum(grad_output_*(-1./X_std.expand_as(X)),/*axis=*/=-1,/*keepdim=*/=True) + grad_sigma*numb*2.0
      *torch::sum((X-X_mean),/*axis=*/=-1,/*keepdim=*/=True)*-1;
  auto grad_x = grad_output_ * (1/X_std.expand_as(X))+grad_sigma*2.0*torch::sum((X-X_mean),/*axis=*/=-1,/*keepdim=*/=True)+grad_mean*numb;
  
  auto grad_scale = torch::sum(grad_output*X_refined,/*axis=*/=-1)
  auto grad_bias = torch::sum(grad_output,/*axis=*/=-1)
                               

  auto grad_weights = X_scaled.t().mm(grad_output);
    
  return {grad_weights, grad_bias, grad_scale, grad_x, };
}


