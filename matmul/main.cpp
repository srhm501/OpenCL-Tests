// Performs sparse matrix - dense vector multiplication

#include <CL/cl.hpp>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

template <typename T>
void print_vec(const std::vector<T> &v)
{
   for (auto &elem : v)
      std::cout << elem << ' ';
   std::cout << std::endl;
}

template <typename T>
size_t get_size(const std::vector<T> &v)
{
   return v.size()*sizeof(v[0]);
}


// Compressed sparse row format matrix
struct CSRmatrix {
   std::vector<double> vals;
   std::vector<int> row_ptr;
   std::vector<int> col_idx;
};

int main(void)
{
   // init host memory
   const CSRmatrix M = {  // [ 0 0 0 0 ]
      {5, 8, 3, 6},       // [ 5 8 0 0 ]
      {0, 0, 2, 3, 4},    // [ 0 0 3 0 ]
      {0, 1, 2, 1}};      // [ 0 6 0 0 ]

   const std::vector<double> vec = {1, 2, 3, 4};
   std::vector<double> ret(vec.size());

   // set up OpenCL platform
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);

   std::vector<cl::Device> devices;
   platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

   cl::Context context(devices);

   cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

   // allocate device memory
   cl::Buffer Mv = cl::Buffer(context, CL_MEM_READ_ONLY, get_size(M.vals));
   cl::Buffer Mr = cl::Buffer(context, CL_MEM_READ_ONLY, get_size(M.row_ptr));
   cl::Buffer Mc = cl::Buffer(context, CL_MEM_READ_ONLY, get_size(M.col_idx));
   cl::Buffer V  = cl::Buffer(context, CL_MEM_READ_ONLY, get_size(vec));
   cl::Buffer R  = cl::Buffer(context, CL_MEM_WRITE_ONLY, get_size(ret));

   // init device memory
   queue.enqueueWriteBuffer(Mv, CL_TRUE, 0, get_size(M.vals), &M.vals[0]);
   queue.enqueueWriteBuffer(Mr, CL_TRUE, 0, get_size(M.row_ptr), &M.row_ptr[0]);
   queue.enqueueWriteBuffer(Mc, CL_TRUE, 0, get_size(M.col_idx), &M.col_idx[0]);
   queue.enqueueWriteBuffer(V, CL_TRUE, 0, get_size(vec), &vec[0]);

   // read in kernel source
   std::ifstream sourceFile("matmul.cl");
   std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),
                          (std::istreambuf_iterator<char>()));
   cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(),
                                                 sourceCode.length()+1));

   cl::Program program = cl::Program(context, source);
   if (program.build(devices) != CL_SUCCESS) {
      std::cerr << "Error building: ";
      std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
      std::cerr << std::endl;
      return EXIT_FAILURE;
   }

   cl::Kernel matmul_kernel(program, "matmul");
   matmul_kernel.setArg(0, Mv);
   matmul_kernel.setArg(1, Mr);
   matmul_kernel.setArg(2, Mc);
   matmul_kernel.setArg(3, V);
   matmul_kernel.setArg(4, R);

   cl::NDRange global(vec.size());
   cl::NDRange local(1);

   queue.enqueueNDRangeKernel(matmul_kernel, cl::NullRange, global, local);
   queue.enqueueReadBuffer(R, CL_TRUE, 0, get_size(ret), &ret[0]);

   print_vec(ret);

   return EXIT_SUCCESS;
}
