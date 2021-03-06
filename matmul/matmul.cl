__kernel
void matmul(__constant double *Mvals,
            __constant uint   *Mrow_ptr,
            __constant uint   *Mcol_idx,
            __constant double *V,
            __global   double *Ret)
{
   const size_t tid = get_global_id(0);

   double sum = 0.0;

   for (uint j=Mrow_ptr[tid]; j<Mrow_ptr[tid+1]; ++j) {
      sum += Mvals[j] * V[Mcol_idx[j]];
   }

   Ret[tid] = sum;
}
