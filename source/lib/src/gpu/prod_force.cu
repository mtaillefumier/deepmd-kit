#include "device.h"
#include "prod_force.h"

template <typename FPTYPE, int THREADS_PER_BLOCK>
__global__ void force_deriv_wrt_center_atom(FPTYPE* force,
                                            const FPTYPE* net_deriv,
                                            const FPTYPE* in_deriv,
                                            const int ndescrpt,
                                            const int nloc,
                                            const int nall) {
  __shared__ FPTYPE data[THREADS_PER_BLOCK * 3];
  int_64 bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  for (int ii = tid; ii < THREADS_PER_BLOCK * 3; ii += THREADS_PER_BLOCK) {
    data[ii] = (FPTYPE)0.;
  }
  for (int ii = tid; ii < ndescrpt; ii += THREADS_PER_BLOCK) {
    for (int jj = 0; jj < 3; jj++) {
      data[jj * THREADS_PER_BLOCK + tid] +=
          net_deriv[bid * ndescrpt + ii] *
          in_deriv[bid * ndescrpt * 3 + ii * 3 + jj];
    }
  }
  __syncthreads();
  // do reduction in shared memory
  for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
    if (tid < ii) {
      for (int jj = 0; jj < 3; jj++) {
        data[jj * THREADS_PER_BLOCK + tid] +=
            data[jj * THREADS_PER_BLOCK + tid + ii];
      }
    }
    __syncthreads();
  }
  // write result for this block to global memory
  const int_64 kk = bid / nloc;  // frame index
  const int_64 ll = bid % nloc;  // atom index
  const int_64 i_idx_nall = kk * nall + ll;
  if (tid == 0) {
    force[i_idx_nall * 3 + 0] -= data[THREADS_PER_BLOCK * 0];
    force[i_idx_nall * 3 + 1] -= data[THREADS_PER_BLOCK * 1];
    force[i_idx_nall * 3 + 2] -= data[THREADS_PER_BLOCK * 2];
  }
}

/* compute $\partial E_j / \partial D_{ji} \nabla_R_j D_{ji}$ */
template <typename FPTYPE, bool radial_only_ = true>
__inline__ __device__ void calculate_forces(
    FPTYPE* forces__,
    const FPTYPE* __restrict__ net_deriv__,
    const FPTYPE* __restrict__ in_deriv__,
    const int idx__) {
  const int offset_j = idx__ * ((radial_only_) ? (1) : (4));
  for (int idw = 0; idw < ((radial_only_) ? (1) : (4)); ++idw) {
    const FPTYPE cst1 = net_deriv__[offset_j + idw];
    forces__[0] += cst1 * in_deriv__[(offset_j + idw) * 3 + 0];
    forces__[1] += cst1 * in_deriv__[(offset_j + idw) * 3 + 1];
    forces__[2] += cst1 * in_deriv__[(offset_j + idw) * 3 + 2];
  }
}

/*
  The original code computes

  f_j += \sum_{k neighbor of j} \delta_{ik} f_jk

  Parallelization is over the i index not the j index which means that several
  threads can update the force of the same atom j. It requires an atomic
  operation to do so.

  Instead each thread should update a unique atom $j$ (basically the $f_j$) and
  computes the corresponding term on the right end side of

  f_j += \sum_{k neighbor of j} f_kj.

  In practice, we associate a unique thread block to each atom $j$ while the block
  threads compute the f_kj (note the reverse order in the indices) individually.
  The final sum (over the neighbors k of the atom j) is a block reduction that
  can be done deterministically.
*/

template <typename FPTYPE, bool radial_only_ = true, int shared_memory_block_>
__global__ void force_deriv_wrt_neighbors(
    FPTYPE* force,
    const FPTYPE* net_deriv,
    const FPTYPE* in_deriv,
    const int* nlist,  // neigbhor list
    const int nframes,
    const int nloc,  // number of atoms on this GPU
    const int nall,  // total number of atoms
    const int nnei)  // number fo neighbors
{
  // limited to 2 billions atoms and 2 billions frames
  const int atom_id = blockIdx.x;
  const int frame_id = blockIdx.z * gridDim.y + blockIdx.y;

  if (frame_id >= nframes) {
    return;
  }

  /*
    structure of the arrays:
       - nlist[] -> nlist[frame][atom][neighbor] (in multi dimensional array
    form)
       - net_deriv[] -> net_deriv[frame][atom][neighbor][1 or 4 elements]
       - in_deriv[] -> in_deriv[frame][atom][neighbor][1 or 4 elements][xyz]
       - force[] -> force[frame][all_atoms][xyz]
  */
  const int ndescrpt = nnei * ((radial_only_) ? (1) : (4));

  // define various pointers for a specific frame.
  const FPTYPE* frame_net_deriv_ =
      &net_deriv[frame_id * nloc *
                 ndescrpt];  // f_net_deriv[atom][neighbor][1 or 4]
  const FPTYPE* frame_in_deriv_ =
      &in_deriv[frame_id * nloc * ndescrpt *
                3];  // f_in_deriv[atom][neighbor][1 or 4][3]
  const int* frame_neighbor_list_ =
      &nlist[frame_id * nnei * nloc];  // f_neighbor_list[atom][neighbors]
  FPTYPE force_tmp[3] = {(FPTYPE)0., (FPTYPE)0., (FPTYPE)0.};

  // used to broadcast if a given atom i has some of its neighbors on this GPU.
  // In that case we may need to apply a final reduction at the end of the
  // computation
  //
  // it is always set to 1 when we use one GPU only.
  __shared__ int do_reduction;
  if (threadIdx.x == 0) {
    do_reduction = ((nall == nloc) ? (1) : (0));
  }
  __syncthreads();

  if (nloc != nall) {
    /*
      Each GPU contains a subpart of the entire system.

      - The neighbor list only contains informations about the atoms located on
      that GPU not the entire system.

      - we do not have all needed information to do the computation so we have
      to rely on the same logic than the original implementation using atomic
      operations.

      - To avoid the atomicAdd we treat each atom one by one and follow these
      steps, (i) search if it is a neighbor of one or several local atoms (data
      are on this GPU), (ii) if so calculate the forces and accumulate the
      results locally (iii) when done apply a final reduction and store the
      results back in global memory
     */

    for (int idx = threadIdx.x; idx < nloc * nnei; idx += blockDim.x) {
      if (frame_neighbor_list_[idx] == atom_id) {
        calculate_forces<FPTYPE, radial_only_>(force_tmp, frame_net_deriv_,
                                               frame_in_deriv_, idx);

        // the atomicAdd has NO impact on determinism. It is used to update a
        // shared memory variable value across the thread block to indicate if
        // we need to apply a block reduction at the end of the calculations.
        // It will be different from zero if at least one thread has all
        // informations for the computations. The value should simply be
        // different from zero and updated once. The worst case scenario will
        // be if all threads of a given block have to do the calculations.

        if (do_reduction == 0) {
          atomicAdd(&do_reduction, 1);
        }
        break;
      }
    }
  } else {
    /* Each GPU has the full information about the system. retrieving
       information is easier as we only have to do a linear search over the
       neighbor list of the neighbors not the entire neighbor list. */

    /* each thread selects a neighbor. */
    for (int neighbor_id = threadIdx.x; neighbor_id < nnei;
         neighbor_id += blockDim.x) {
      // collect all terms $\partial E_j / \partial D_{ji} \nabla_R_j D_{ji}$
      // where the atom i is a neighbor of the atom j.
      //
      // Go through all neighbors of atom i, locate the position of
      // the atom i in the neighbor list of the atom j and retrieve all
      // necessary information.

      const int atom_j = frame_neighbor_list_[atom_id * nnei + neighbor_id];

      // The neighbors of a given atom are sorted by type and each resulting
      // list is separated from the other by a series of -1. More details about
      // the sorting can be found in https://doi.org/10.1016/j.cpc.2020.107624
      //
      // To illustrate this, take the neigbhors of a given atom of type a (in a
      // system with two atoms type a and b) deepmd stores the neighbors as
      //
      // [neighbors list of type a], -1, -1, -1, ...., [neighbor list of type
      // b], -1, -1, -1, .....

      if (atom_j < 0) {
        continue;
      }

      const int* nei_nei_list_ = &frame_neighbor_list_[atom_j * nnei];
      int atom_id_position = 0;

      // search the index of the atom i in the local neighbor list of atom j
      for (atom_id_position = 0; atom_id_position < nnei; atom_id_position++) {
        if (nei_nei_list_[atom_id_position] == atom_id) {
          break;
        }
      }
      calculate_forces<FPTYPE, radial_only_>(force_tmp, frame_net_deriv_,
                                             frame_in_deriv_, atom_j * nnei + atom_id_position);
    }
  }

  __syncthreads();

  // Apply the final reduction.
  if (do_reduction) {
    __shared__ FPTYPE fx[shared_memory_block_];
    __shared__ FPTYPE fy[shared_memory_block_];
    __shared__ FPTYPE fz[shared_memory_block_];

    fx[threadIdx.x] = force_tmp[0];
    fy[threadIdx.x] = force_tmp[1];
    fz[threadIdx.x] = force_tmp[2];
    __syncthreads();

    // do the final reduction
    for (int tt = shared_memory_block_ / 2; tt > 0; tt >>= 1) {
      if (threadIdx.x < tt) {
        fx[threadIdx.x] += fx[threadIdx.x + tt];
        fy[threadIdx.x] += fy[threadIdx.x + tt];
        fz[threadIdx.x] += fz[threadIdx.x + tt];
      }
      __syncthreads();
    }

    /* Note the sign difference between the formula in the PRL paper and the
       code. it is due to \nabla_R_j D_{ji} = -\nabla_R_i D_{ji} */
    if (threadIdx.x == 0) {
      const int64_t offset = (frame_id * nall + atom_id) * 3;
      force[offset] += fx[0];
      force[offset + 1] += fy[0];
      force[offset + 2] += fz[0];
    }
  }
}

template <typename FPTYPE, bool radial_only_ = true>
void prod_force_a_r_gpu(FPTYPE* force,
                        const FPTYPE* net_deriv,
                        const FPTYPE* in_deriv,
                        const int* nlist,
                        const int nloc,
                        const int nall,
                        const int nnei,
                        const int nframes) {
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());

  const int ndescrpt = nnei * ((radial_only_) ? (1) : (4));
  DPErrcheck(gpuMemset(force, 0, sizeof(FPTYPE) * nframes * nall * 3));

  force_deriv_wrt_center_atom<FPTYPE, TPB><<<nframes * nloc, TPB>>>(
      force, net_deriv, in_deriv, ndescrpt, nloc, nall);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());

  dim3 block_grid;

  if (nframes < 0xffff) {
    block_grid = dim3(nall, nframes, 1);
  } else {
    const int sqrt_nframes = sqrt(nframes);
    block_grid = dim3(nall, sqrt_nframes + 1, sqrt_nframes + 1);
  }
  // to accomodate AMD GPU
  dim3 thread_grid(64, 1, 1);
  force_deriv_wrt_neighbors<FPTYPE, radial_only_, 64>
      <<<block_grid, thread_grid>>>(force, net_deriv, in_deriv, nlist, nframes,
                                    nloc, nall, nnei);
  DPErrcheck(gpuGetLastError());
  DPErrcheck(gpuDeviceSynchronize());
}

namespace deepmd {
template <typename FPTYPE>
void prod_force_a_gpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes) {
  prod_force_a_r_gpu<FPTYPE, false>(force, net_deriv, in_deriv, nlist, nloc,
                                    nall, nnei, nframes);
}

template <typename FPTYPE>
void prod_force_r_gpu(FPTYPE* force,
                      const FPTYPE* net_deriv,
                      const FPTYPE* in_deriv,
                      const int* nlist,
                      const int nloc,
                      const int nall,
                      const int nnei,
                      const int nframes) {
  prod_force_a_r_gpu<FPTYPE, true>(force, net_deriv, in_deriv, nlist, nloc,
                                   nall, nnei, nframes);
}

template void prod_force_a_gpu<float>(float* force,
                                      const float* net_deriv,
                                      const float* in_deriv,
                                      const int* nlist,
                                      const int nloc,
                                      const int nall,
                                      const int nnei,
                                      const int nframes);
template void prod_force_a_gpu<double>(double* force,
                                       const double* net_deriv,
                                       const double* in_deriv,
                                       const int* nlist,
                                       const int nloc,
                                       const int nall,
                                       const int nnei,
                                       const int nframes);
template void prod_force_r_gpu<float>(float* force,
                                      const float* net_deriv,
                                      const float* in_deriv,
                                      const int* nlist,
                                      const int nloc,
                                      const int nall,
                                      const int nnei,
                                      const int nframes);
template void prod_force_r_gpu<double>(double* force,
                                       const double* net_deriv,
                                       const double* in_deriv,
                                       const int* nlist,
                                       const int nloc,
                                       const int nall,
                                       const int nnei,
                                       const int nframes);
}  // namespace deepmd
