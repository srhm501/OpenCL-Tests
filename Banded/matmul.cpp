#include <iostream>
#include <vector>

#define MIN(x,y) ((x<y)?x:y)

template <typename T>
class DIAmatrix
{
   std::vector<int> m_diags;
   std::vector<T> m_vals;
   unsigned m_bandwidth;
   unsigned m_n;
   unsigned m_m;
   unsigned m_max_diag;

public:

   DIAmatrix(const std::vector<int> &diags,
             const std::vector<T> &vals,
             const unsigned bandwidth,
             const unsigned n,
             const unsigned m) :
      m_diags(diags),
      m_vals(vals),
      m_bandwidth(bandwidth),
      m_n(n),
      m_m(m),
      m_max_diag(MIN(m,n))
   {
   }

   T at(const int r, const int c) const
   {
      const int diag = c-r;

      unsigned i;
      for (i=0; i<m_diags.size(); ++i)
         if (diag == m_diags[i])
            return m_vals[i*m_max_diag + r];

      return 0;
   }

   std::vector<T> operator*(const std::vector<T> &v) const
   {
      std::vector<T> ret(v.size(), 0);

      for (unsigned i=0; i<v.size(); ++i)
         for (unsigned j=0; j<v.size(); ++j)
         {
            std::cout << i << ' ' << j << ' ' << this->at(i,j) << std::endl;
            ret[i] += this->at(i,j)*v[j];
         }

      return ret;
   }
};


int main(void)
{
   DIAmatrix<double> M({-1, 0, 1}, {0, 3, 3, 3, 1,1,1,1, 2, 2, 2, 0}, 3, 4, 4);
   std::vector<double> v = {1, 2, 3, 4};

   std::vector<double> r = M*v;
   
   for (auto &elem : r)
      std::cout << elem << ' ';
   std::cout << std::endl;
}
