#include <iostream>
#include <cstdint>

const uint32_t w = 32;
const uint32_t n = 624;
const uint32_t m = 397;
const uint32_t r = 31;
const uint32_t a = 0x9908B0DF;
const uint32_t f = 1812433253;
const uint32_t u = 11;
const uint32_t d = 0xFFFFFFFF;
const uint32_t s = 7;
const uint32_t b = 0x9D2C5680;
const uint32_t t = 15;
const uint32_t c = 0xEFC60000;
const uint32_t l = 18;
const uint32_t lower_mask = (1ull << r) - 1;
const uint32_t upper_mask = (~lower_mask) & ((1ull << w) - 1);

uint32_t MT[n];
uint32_t index = n+1;

void seed_mt(const uint32_t seed)
{
   index = n;
   MT[0] = seed;
   for (uint32_t i=1; i<n; ++i) {
      MT[i] = (f * (MT[i-1] ^ (MT[i-1] >> (w-2))) + i);
   }
}

void twist()
{
   for (uint32_t i=0; i<n; ++i) {
      uint32_t x = (MT[i] & upper_mask) + (MT[(i+1)%n] & lower_mask);
      uint32_t xA = x >> 1;
      if (x & 0x1) {
         xA ^= a;
      }
      MT[i] = MT[(i+m)%n] ^ xA;
   }
   index = 0;
}

uint32_t extract_number()
{
   if (index >= n) {
      if (index > n) {
         std::cerr << "Generator not seeded" << std::endl;;
      }

      twist();
   }

   uint32_t y = MT[index];
   y ^= ((y >> u) & d);
   y ^= ((y << s) & b);
   y ^= ((y << t) & c);
   y ^= (y >> l);

   ++index;

   return y & ((1ull << w) - 1);
}

int main(void)
{
   seed_mt(5489);
   for (int i=0; i<10; ++i)
      std::cout << extract_number()/double(0xFFFFFFFF) << std::endl;
}
