__kernel
void matmul(__constant double *Mvals,
            __constant unsigned int *Mrow_ptr,
            __constant unsigned int *Mcol_idx,
            __constant double *V,
            __global   double *Ret)
{
   const unsigned int tid = get_global_id(0);

   Ret[tid] = 0.0;

   for (unsigned int j=Mrow_ptr[tid]; j<Mrow_ptr[tid+1]; ++j) {
      Ret[tid] += Mvals[j] * V[Mcol_idx[j]];
   }
}
