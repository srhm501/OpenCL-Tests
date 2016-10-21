__kernel
void matmul(__constant double *Mvals,
            __constant int    *Mrow_ptr,
            __constant int    *Mcol_idx,
            __constant double *V,
            __global   double *Ret)
{
   const size_t tid = get_global_id(0);

   Ret[tid] = 0.0;

   for (size_t j=Mrow_ptr[tid]; j<Mrow_ptr[tid+1]; ++j) {
      Ret[tid] += Mvals[j] * V[Mcol_idx[j]];
   }
}
