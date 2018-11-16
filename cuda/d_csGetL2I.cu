/* Copyright (C) 2018 Jack Ogaja.
*
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
*
*  The above copyright notice and this permission notice shall be included in all
*  copies or substantial portions of the Software.
*
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*  SOFTWARE.
*/

/*--functions for reading ESA's CryoSat2 data sets on CUDA GPUs--*/

/*==============
These functions use I/O libraries developed by UCL/MSSL, London
I/O library author: (C) CryoSat2 Software Team, 
                   Mullard Space Science Lab, UCL, London
I/O library version: 2.3
================*/

/*
Jack Ogaja, 
Brown University,
20180620
See LICENSE.md for copyright notice
*/

#include "csarray.h"
#include "cuda.h"

// code to be executed by each CUDA "thread"
__global__ void 
readAndFillKernel(unsigned long long int n_skip, 
                                   long int y_f, 
                       unsigned long long int N, 
                                 L2IData* l_data)
{  

  /** Loop counter for file reading */
  unsigned long long int i = 0;

  /** The index of the current record being read */
  unsigned long long int j = 0;

  int threadRank = threadIdx.x; // thread rank in thread-block
  int blockRank = blockIdx.x;   // rank of thread-block
  int blockSize = blockDim.x;   // number of threads in each thread-block

  int i = threadRank + blockSize*blockRank;

  /* Read and fill */
  if( (i>=n_skip) && (i<N) )
  {
      j = i - n_skip;

      y_f  = jFillL2IStructFromFile(&l_data[ j ],
                                    t_handle->pt_filepointer , t_handle->j_baseline);

      if( y_f != EXIT_SUCCESS ) return NULL; 

  }

}

//main procedure
L2IData* 
d_csGetL2I(t_cs_filehandle   h_t_handle,
           long int               d_i,
           unsigned long long int d_N,
           unsigned long long int d_n,
           L2IData*            h_data )
{
  /** Total number of records to read from the file */
  unsigned long long int tot_n = 0;

  /** Return value of function calls */
  long int y_f = 0;

  if( h_t_handle == NULL ) return NULL; 

  if( h_t_handle->pt_filepointer == NULL ) return NULL; 

  if( h_t_handle->j_num_datasets <= j_dataset_index ) return NULL; 

  if( h_t_handle->j_type != CS_L2I_DS_NAME_ANY )
  {
       printf( "File is not an L2I filetype.\n" );
       return NULL;
  }

  if( h_data == NULL )
  {
      if( d_N == 0 )
      {
          /* Determine number of records in entire file */
          d_N = h_t_handle->paj_num_records[ d_i ];
      }

      if( d_n != 0 )
      {
          tot_n = d_N - d_n;
      }
      else
      {
          tot_n = d_N;
      }

      h_data = malloc( sizeof( L2IData ) * tot_n );

      if( h_data == NULL ) return NULL; 
    }

  /* Put the filepointer at the start of the data */
  y_f = fsetpos( h_t_handle->pt_filepointer,
                      &h_t_handle->pat_data_start_offsets[ d_i ] );

  if( y_f != EXIT_SUCCESS ) return NULL; 

  // Copy data to the device
  L2IData* d_data;
  t_cs_filehandle  d_t_handle;

  cudaMalloc(&d_data, tot_n*sizeof(L2IData));
  cudaMalloc(&d_t_handle, sizeof(t_cs_filehandle));

  cudaMemcpy(d_data, h_data, tot_n*sizeof(L2IData), cudaMemcpyHostToDevice);
  cudaMemcpy(d_t_handle, h_t_handle, sizeof(t_cs_filehandle), cudaMemcpyHostToDevice);

  int TPB = 100;
  int B = (N + TPB -1)/TPB;

  // execute the kernel code with TPB threads per block and B thread-blocks
  // (total of B*TPB threads)
  readAndFillKernel <<< B, TPB >>> (d_n, y_f, d_N, d_data );

  // Copy the data back to the host
  cudaMemcpy(h_data, d_data, N*sizeof(double), cudaMemcpyDeviceToHost);

  // Free the device memory
  cudaFree(d_data);
  cudaFree(d_t_handle);

  return h_data;

} /* d_CSGetL2I */
