cmake_minimum_required(VERSION 3.10)
project(pytorch_v4)

set(CMAKE_CXX_STANDARD 14)

add_executable(pytorch_v4 main.cpp tensor.hpp node.hpp module.hpp optimizer.hpp loss.hpp torch.hpp activate.hpp utils.hpp)
