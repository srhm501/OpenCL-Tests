#include <CL/cl.hpp>
#include <vector>
#include <iostream>

int main(void)
{
   size_t nplatforms = 0;
   size_t ndevices = 0;
   
   std::vector<cl::Platform> platforms;
   cl::Platform::get(&platforms);
   nplatforms = platforms.size();

   std::vector<std::vector<cl::Device>> devices(nplatforms);
   for (size_t i=0; i<nplatforms; ++i)
   {
      std::vector<cl::Device> tmp_devices;
      platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &tmp_devices);
      devices[i] = tmp_devices;
      ndevices += tmp_devices.size();
   }

   std::cout << "Found " << nplatforms << " platform(s), with " << ndevices << " device(s)" << std::endl;

   for (size_t i=0; i<nplatforms; ++i)
   {
      std::cout << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
      for (size_t j=0; j<devices[i].size(); ++j)
      {
         std::cout << '\t' << devices[i][j].getInfo<CL_DEVICE_NAME>() << std::endl;
      }
   }

   return 0;
}
