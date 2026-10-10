// SPDX-License-Identifier: MIT
#include "mini_llama/ops.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mini_llama{

namespace{

void CheckSameShape(const Tensor& a, const Tensor&b, const char* caller){
    if(a.shape!=b.shape){
        throw std::runtime_error(std::string(caller)+": shape mismatch " +
                             a.ShapeStringShort() + " vs " +
                             b.ShapeStringShort());
    }    
}
float SiluScalar(float x){ return x/(1.0f+std::exp(-x));};

} // namespace

Tensor Matmul(const Tensor&a, const Tensor&b){
    if(a.num_dims()!=2 || b.num_dims()!=2){
        throw std::runtime_error("Matmul: expected 2D tensors");
    }
    if(a.shape[1]!=b.shape[0]){
        throw std::runtime_error("Matmul: inner dimension mismatch " +
                             a.ShapeStringShort() + " x " +
                             b.ShapeStringShort());
    }
    const int m=a.shape[0];
    const int k=a.shape[1];
    const int n=b.shape[1];
    Tensor c({m,n},0.0f);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            float sum=0.0f;
            for(int t=0;t<k;t++){
                sum+=a.data[i*k+t]*b.data[t*n+j];
            }
            c.data[i*n+j]=sum;
        }
    }
    return c;
}

// y=x * weight^T，全连接层，x可以理解为行向量有in个，weight=[out,in]，输出
Tensor Linear(const Tensor&x, const Tensor& weight){
    if(weight.num_dims()!=2){
        throw std::runtime_error("Linear: expected weight [out, in]");
    }
    int in_features=0;
    if(x.num_dims()==1){
        in_features=x.shape[0];
    }else if(x.num_dims()==2&&x.shape[0]==1){
        in_features=x.shape[1];
    }else{
        throw std::runtime_error("Linear: expected x [in] or [1, in]");
    }
    if(weight.shape[1]!=in_features){
        throw std::runtime_error("Linear: dimension mismatch, x " +
                             x.ShapeStringShort() + ", weight " +
                             weight.ShapeStringShort());
    }
    const int out_features=weight.shape[0];
    Tensor y =x.num_dims()==1? Tensor({out_features},0.0f) : 
                            Tensor({1,out_features},0.0f);
    for(int o=0;o<out_features;o++){
        const float* w_row=weight.RowPtr(o);
        float sum=0.0f;
        for(int i=0;i<in_features;i++){
            sum+=w_row[i]*x.data[i];
        }
        y.data[o]=sum;
    }
    return y;
}

// 均方根归一化
// y = x /sqrt(mean(x^2)+eps) * weight;
Tensor RmsNorm(const Tensor&x, const Tensor& weight, float eps){
    if(x.num_dims()!=1){
        throw std::runtime_error("RmsNorm: expected 1D x");
    }
    CheckSameShape(x,weight,"RmsNorm");
    float sum_sq=0.0f;
    for(float v:x.data){
        sum_sq+=v*v;
    }
    const float mean_sq=sum_sq/static_cast<float>(x.size());
    const float scale=1.0f/std::sqrt(mean_sq+eps);

    Tensor y(x.shape,0.0f);
    for(int i=0;i<x.size();i++){
        y.data[i]=x.data[i]*scale*weight[i];
    }
    return y;
}

// Softmax: 把数值分布转化成概率分布，这里要把每个数都减去exp(最大数)
Tensor Softmax(const Tensor& x){
    if(x.num_dims()!=1){
        throw std::runtime_error("Softmax: expected 1D tensor, got " +
                             x.ShapeStringShort());
    }
    if(x.data.empty()){
        throw std::runtime_error("Softmax: empty tensor");
    }
    float max_val=x.data[0];
    for(float v:x.data){
        if(v>max_val)
            max_val=v;
    }

    Tensor y(x.shape,0.0f);
    float sum=0.0f;
    for(int i=0;i<x.size();i++){
        y.data[i]=std::exp(x.data[i]-max_val);
        sum+=y.data[i];
    }
    for(float &v:y.data){
        v/=sum;
    }
    return y;
}

Tensor Silu(const Tensor& x){
    Tensor y =x;
    for(float& v:y.data){
        v=SiluScalar(v);
    }
    return y;
}

Tensor ElementwiseMul(const Tensor&a, const Tensor& b){
    CheckSameShape(a,b,"ElementwiseMul");
    Tensor y(a.shape,0.0f);
    for(int i=0;i<a.size();i++){
        y.data[i]=a.data[i]*b.data[i];
    }
    return y;
}

Tensor SwiGlu(const Tensor& gate, const Tensor& up){
    CheckSameShape(gate,up,"SwiGlu");
    Tensor y(gate.shape,0.0f);
    for(int i=0;i<gate.size();i++){
        y.data[i]=SiluScalar(gate.data[i])*up[i];
    }
    return y;
}

// 这个函数是用来处理token的位置信息的。实现起来比较复杂
void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    if (q.num_dims() != 2 || k.num_dims() != 2) {
      throw std::runtime_error("Rope: expected q/k [n_heads, head_dim]");
    }
    const int head_dim = q.shape[1];
    if (k.shape[1] != head_dim) {
      throw std::runtime_error("Rope: head_dim mismatch, q " +
                               q.ShapeStringShort() + ", k " +
                               k.ShapeStringShort());
    }
    if (head_dim % 2 != 0) {
      throw std::runtime_error("Rope: head_dim must be even, got " +
                               std::to_string(head_dim));
    }
    if (pos < 0) {
      throw std::out_of_range("Rope: position must be non-negative, got " +
                              std::to_string(pos));
    }
    if (!(theta > 0.0f)) {
      throw std::runtime_error("Rope: theta must be positive");
    }
    const int half = head_dim / 2;
    auto rotate = [&](Tensor& t) {
      for (int h = 0; h < t.shape[0]; ++h) {
        float* row = t.RowPtr(h);
        for (int i = 0; i < half; ++i) {
          const float freq =
              std::pow(theta, -2.0f * static_cast<float>(i) / head_dim);
          const float angle = static_cast<float>(pos) * freq;
          const float c = std::cos(angle);
          const float s = std::sin(angle);
          // kNormal pairs (2i, 2i+1); kNeoX pairs (i, i + half)
          const int i0 = rope_type == RopeType::kNeoX ? i : 2 * i;
          const int i1 = rope_type == RopeType::kNeoX ? i + half : 2 * i + 1;
          const float x0 = row[i0];
          const float x1 = row[i1];
          row[i0] = x0 * c - x1 * s;
          row[i1] = x0 * s + x1 * c;
        }
      }
    };
    rotate(q);
    rotate(k);
}

int ArgMax(const Tensor& x){
    if(x.data.empty()){
        throw std::runtime_error("ArgMax: empty tensor");
    }
    int best=0;
    for(int i=1;i<x.size();i++){
        if(x.data[i]>x.data[best]){
            best=i;
        }
    }
    return best;
}
    
}// namespace mini_llama