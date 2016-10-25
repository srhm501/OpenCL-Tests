#include <CL/cl.hpp>
#include <vector>
#include <iostream>

int main(void)
{
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);

   std::vector<std::vector<cl::Device>> devices(platforms.size());
   for (size_t i=0; i<platforms.size(); ++i)
   {
      std::vector<cl::Device> tmp_devices;
      platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &tmp_devices);
      devices[i] = tmp_devices;
   }

   for (size_t i=0; i<platforms.size(); ++i)
   {
      std::cout << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
      for (size_t j=0; j<devices[i].size(); ++j)
      {
         std::cout << '\t' << devices[i][j].getInfo<CL_DEVICE_NAME>() << std::endl;
      }
   }

   return 0;
}
