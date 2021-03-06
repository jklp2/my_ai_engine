cmake_minimum_required(VERSION 3.17)
project(pytorch_v4)

set(CMAKE_CXX_STANDARD 14)

add_executable(pytorch_v4 main.cpp tensor.h node.h module.h optimizer.h loss.h torch.h activate.h utils.h)