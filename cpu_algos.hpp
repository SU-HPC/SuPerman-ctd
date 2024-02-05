#ifndef ALGO_HPP
#define ALGO_HPP

#include <iostream>
#include <bitset>
#include <omp.h>
#include <string.h>
#include "flags.h"
#include "util.h"
#include <typeinfo>
using namespace std;


template <class T>
double greedy(T* mat, int nov, int number_of_times) {
  T* mat_t = new T[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }
  
  srand(time(0));
  
  int nt = omp_get_max_threads();
  T sum_perm = 0;
  long long int sum_zeros = 0;
  
#pragma omp parallel for num_threads(nt) reduction(+:sum_perm) reduction(+:sum_zeros)
  for (int time = 0; time < number_of_times; time++) {
    float row_nnz[nov];
    float col_nnz[nov];
    bool row_extracted[nov];
    bool col_extracted[nov];
    
    for (int i = 0; i < nov; i++) {
      row_nnz[i] = 0;
      col_nnz[i] = 0;
      row_extracted[i] = false;
      col_extracted[i] = false;
      for (int j = 0; j < nov; j++) {
	if (mat[(i * nov) + j] != 0) {
	  row_nnz[i] += 1;
	}
	if (mat_t[(i * nov) + j] != 0) {
	  col_nnz[i] += 1;
	}
      }
    }
    
    double perm = 1;
    int row, col, deg;
    float sum_pk, pk, random;
    
    for (int i = 0; i < nov; i++) {
      // choose the row to be extracted
      deg = nov+1;
      int n = 0;
      for (int l = 0; l < nov; l++) {
	if (!row_extracted[l]) {
	  if (row_nnz[l] < deg) {
	    n = 1;
	    row = l;
	    deg = row_nnz[l];
	  } else if (row_nnz[l] == deg) {
	    n++;
	  }
	}
      }
      if (n > 1) {
	random = rand() % n;
	for (int l = 0; l < nov; l++) {
	  if (!row_extracted[l] && row_nnz[l] == deg) {
	    if (random == 0) {
	      row = l; break;
	    }
	    random--;
	  }
	}
      }
      
      // compute sum of probabilities, when finding deg = 1, set col
      sum_pk = 0;
      int sum_ones = 0;
      for (int k = 0; k < nov; k++) {
	if (!col_extracted[k] && mat[row * nov + k] != 0) {
	  sum_pk += 1 / col_nnz[k];
	}
      }
      
      // choose the col to be extracted if not chosen already
      if (sum_ones == 0) {
	random = (float(rand()) / RAND_MAX) * sum_pk;
	sum_pk = 0;
	for (int k = 0; k < nov; k++) {
	  if (!col_extracted[k] && mat[row * nov + k] != 0) {
	    sum_pk += 1 / col_nnz[k];
	    if (random <= sum_pk) {
	      col = k;
	      pk = 1 / col_nnz[k];
	      break;
	    }
	  }
	}
      } else {
	random = rand() % sum_ones;
	for (int k = 0; k < nov; k++) {
	  if (!col_extracted[k] && mat[row * nov + k] != 0 && col_nnz[k] == 1) {
	    if (random == 0) {
	      col = k;
	      pk = 1;
	      break;
	    }
	    random--;
	  }
	}
      }
      
      // multiply permanent with 1 / pk
      perm /= pk;
      
      // extract row and col
      row_extracted[row] = true;
      col_extracted[col] = true;
      
      // update number of nonzeros of the rows and cols after extraction
      bool zero_row = false;
      for (int r = 0; r < nov; r++) {
	if (!row_extracted[r] && mat_t[col * nov + r] != 0) {
	  row_nnz[r]--;
	  if (row_nnz[r] == 0) {
	    zero_row = true;
	    break;
	  }
	}
      }
      
      if (zero_row) {
	perm = 0;
	sum_zeros += 1;
	break;
      }
      
      for (int c = 0; c < nov; c++) {
	if (!col_extracted[c] && mat[row * nov + c] != 0) {
	  col_nnz[c]--;
	  if (col_nnz[c] == 0) {
	    zero_row = true;
	    break;
	  }
	}
      }
      
      if (zero_row) {
	perm = 0;
	sum_zeros += 1;
	break;
      }
    }
    
    sum_perm += perm;
  }
  
  delete[] mat_t;
  
  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

template <class C, class S>
Result rasmussen_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags) {
  
  //Pack parameters
  S* mat = densemat->mat;
  int* rptrs = sparsemat->rptrs;
  int* cols = sparsemat->cols;
  int nov = sparsemat->nov;
  //Pack parameters
  
  //Pack flags//
  int number_of_times = flags.number_of_times;
  int threads = flags.threads;
  //Pack flags//
  
  double starttime = omp_get_wtime();
  
  S* mat_t = new S[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }
  
  srand(time(0));
  
  C sum_perm = 0;
  long long int sum_zeros = 0;
  
#pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
  for (int time = 0; time < number_of_times; time++) {
    int row_nnz[nov];
    bool col_extracted[nov];
    bool row_extracted[nov];
    for (int i = 0; i < nov; i++) {
      col_extracted[i] = false;
      row_extracted[i] = false;
    }
    
    int row;
    int min = nov+1;
    
    for (int i = 0; i < nov; i++) {
      row_nnz[i] = rptrs[i+1] - rptrs[i];
      if (min > row_nnz[i]) {
	min = row_nnz[i];
	row = i;
      }
    }
    
    C perm = 1;
    
    for (int k = 0; k < nov; k++) {
      // multiply permanent with number of nonzeros in the current row
      perm *= row_nnz[row];
      
      // choose the column to be extracted randomly
      int random = rand() % row_nnz[row];
      int col;
      for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
	int c = cols[i];
	if (!(col_extracted[c])) {
	  if (random == 0) {
	    col = c;
	    break;
	  } else {
	    random--;
	  }        
	}
      }
      
      // exract the column
      col_extracted[col] = true;
      row_extracted[row] = true; 
      
      min = nov+1;
      
      // update number of nonzeros of the rows after extracting the column
      bool zero_row = false;
      for (int r = 0; r < nov; r++) {
	if (!(row_extracted[r])) {
	  if (mat_t[col * nov + r] != 0) {
	    row_nnz[r]--;
	    if (row_nnz[r] == 0) {
	      zero_row = true;
	      break;
	    }
	  }
	  if (min > row_nnz[r]) {
	    min = row_nnz[r];
	    row = r;
	  }
	}
      }
      
      if (zero_row) {
	perm = 0;
	sum_zeros += 1;
	break;
      }
    }
    
    sum_perm += perm;
  }
  
  delete[] mat_t;
  
  double duration = omp_get_wtime() - starttime;  
  double perman = sum_perm / number_of_times;
  Result result(perman, duration);
  return result;
}


template <class C, class S>
Result rasmussen(DenseMatrix<S>* densemat, flags flags) {

  //Pack parameters//
  S* mat = densemat->mat;
  int nov = densemat->nov;
  //Pack parameters//

  //Pack flags//
  int threads = flags.threads;
  int number_of_times = flags.number_of_times;
  //Pack flags//

  double starttime = omp_get_wtime();

  S* mat_t = new S[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }

  srand(time(0));

  C sum_perm = 0;
  C sum_zeros = 0;
  
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
  for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      long col_extracted = 0;
      long row_extracted = 0;
      
      int row;
      int min = nov+1;

      for (int i = 0; i < nov; i++) {
        row_nnz[i] = 0;
        for (int j = 0; j < nov; j++) {
          if (mat[(i * nov) + j] != 0) {
            row_nnz[i] += 1;
          }
        }
        if (min > row_nnz[i]) {
          min = row_nnz[i];
          row = i;
        }
      }
      
      C perm = 1;
      
      for (int i = 0; i < nov; i++) {
        // multiply permanent with number of nonzeros in the current row
        perm *= row_nnz[row];

        // choose the column to be extracted randomly
        int random = rand() % row_nnz[row];
        int col;
        for (int c = 0; c < nov; c++) {
          if (!((col_extracted >> c) & 1L) && mat[row * nov + c] != 0) {
            if (random == 0) {
              col = c;
              break;
            } else {
              random--;
            }        
          }
        }

        // exract the column
        col_extracted |= (1L << col);
        row_extracted |= (1L << row);

        min = nov+1;
        // update number of nonzeros of the rows after extracting the column
        bool zero_row = false;
        for (int r = 0; r < nov; r++) {
          if (!((row_extracted >> r) & 1L)) {
            if (mat_t[col * nov + r] != 0) {
              row_nnz[r]--;
              if (row_nnz[r] == 0) {
                zero_row = true;
                break;
              }
            }
            if (min > row_nnz[r]) {
              min = row_nnz[r];
              row = r;
            }
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }
      }

      sum_perm += perm;
    }

  delete[] mat_t;

  
  double duration = omp_get_wtime() - starttime;  
  double perman = sum_perm / number_of_times;
  Result result(perman, duration);
  return result;
}

template <class C, class S>
Result approximation_perman64_sparse(SparseMatrix<S>* sparsemat, flags flags) {

  //Pack parameters//
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  int* rptrs = sparsemat->rptrs;
  int* cols = sparsemat->cols;
  int nov = sparsemat->nov;
  //Pack parameters//
  
  //Pack flags//
  int number_of_times = flags.number_of_times;
  int scale_intervals = flags.scale_intervals;
  int scale_times = flags.scale_times;
  int threads = flags.threads;
  //Pack flags//

  double starttime = omp_get_wtime();

  srand(time(0));

  C sum_perm = 0;
  C sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int col_extracted[64]; //??
      int row_extracted[64]; //??
      for (int i = 0; i < 64; i++) {
        col_extracted[i]=0;
        row_extracted[i]=0;
      }

      C Xa = 1;
      C d_r[nov];
      C d_c[nov];
      for (int i = 0; i < nov; i++) {
        d_r[i] = 1;
        d_c[i] = 1;
      }

      int row;
      int min;
      int nnz;

      for (int k = 0; k < nov; k++) {
        min=nov+1;
        for (int i = 0; i < nov; i++) {
          if (!((row_extracted[i / 32] >> (i % 32)) & 1)) {
            nnz = 0;
            for (int j = rptrs[i]; j < rptrs[i+1]; j++) {
              int c = cols[j];
              if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
                nnz++;
              }
            }
            if (min > nnz) {
              min = nnz;
              row = i;
            }
          }
        }

        // Scale part
        if (time % scale_intervals == 0) {
          bool success = ScaleMatrix_sparse(cptrs, rows, rptrs, cols, nov, row_extracted, col_extracted, d_r, d_c, scale_times);
          if (!success) {
            Xa = 0;
            sum_zeros++;
            break;
          }
        }

        // use scaled matrix for pj
        C sum_row_of_S = 0;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
            sum_row_of_S += d_r[row] * d_c[c];
          }
        }
        if (sum_row_of_S == 0) {
          Xa = 0;
          sum_zeros++;
          break;
        }

        C random = (C(rand()) / RAND_MAX) * sum_row_of_S;
        C temp = 0;
        C s, pj;
        int col;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
            s = d_r[row] * d_c[c];
            temp += s;
            if (random <= temp) {
              col = c;
              pj = s / sum_row_of_S;
              break;
            }
          }
        }

        // update Xa
        Xa /= pj;
        
        // exract the column
        col_extracted[col / 32] |= (1 << (col % 32));
        // exract the row
        row_extracted[row / 32] |= (1 << (row % 32));

      }

      sum_perm += Xa;
    }

    double duration = omp_get_wtime() - starttime;  
    double perman = sum_perm / number_of_times;
    Result result(perman, duration);
    return result;
}

template <class C, class S>
Result approximation_perman64(DenseMatrix<S>* densemat, flags flags) {

  //Pack parameters//
  S* mat = densemat->mat;
  int nov = densemat->nov;
  //Pack parameters//
  
  //Pack flags//
  int number_of_times = flags.number_of_times;
  int scale_intervals  = flags.scale_intervals;
  int scale_times = flags.scale_times;
  int threads = flags.threads;
  //Pack flags//

  double starttime = omp_get_wtime();
  
  srand(time(0));

  C sum_perm = 0;
  C sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      long col_extracted = 0;
      long row_extracted = 0;

      C Xa = 1;
      C d_r[nov];
      C d_c[nov];
      for (int i = 0; i < nov; i++) {
        d_r[i] = 1;
        d_c[i] = 1;
      }

      int row;
      int min;
      int nnz;

      for (int k = 0; k < nov; k++) {
        min=nov+1;
        for (int i = 0; i < nov; i++) {
          if (!((row_extracted >> i) & 1L)) {
            nnz = 0;
            for (int j = 0; j < nov; j++) {
              if (!((col_extracted >> j) & 1L) && mat[(i * nov) + j] != 0) {
                nnz++;
              }
            }
            if (min > nnz) {
              min = nnz;
              row = i;
            }
          }
        }
        // Scale part
        if (time % scale_intervals == 0) {
          bool success = ScaleMatrix(mat, nov, row_extracted, col_extracted, d_r, d_c, scale_times);
          if (!success) {
            Xa = 0;
            sum_zeros++;
            break;
          }
        }
        
        // use scaled matrix for pj
        C sum_row_of_S = 0;
        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L) && mat[(row * nov) + j] != 0) {
            sum_row_of_S += d_r[row] * mat[(row * nov) + j] * d_c[j];
          }
        }
        if (sum_row_of_S == 0) {
          Xa = 0;
          sum_zeros++;
          break;
        }
	
        C random = (C(rand()) / RAND_MAX) * sum_row_of_S;
        C temp = 0;
        C s, pj;
        int col;
        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L) && mat[(row * nov) + j] != 0) {
            s = d_r[row] * mat[(row * nov) + j] * d_c[j];
            temp += s;
            if (random <= temp) {
              col = j;
              pj = s / sum_row_of_S;
              break;
            }
          }
        }

        // update Xa
        Xa /= pj;
        
        // exract the column
        col_extracted |= (1L << col);
        // exract the row
        row_extracted |= (1L << row);

      }

      sum_perm += Xa;
    }
    
    double duration = omp_get_wtime() - starttime;  
    double perman = sum_perm / number_of_times;
    Result result(perman, duration);
    return result;
}

template <class C, class S>
Result parallel_perman64_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags) {
  
  //Pack parameters//
  S* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters//
  
  //Pack flags//
  int threads = flags.threads;
  //Pack flags//

  double starttime = omp_get_wtime();
  
  C x[nov];   
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'

#ifdef DTYPE
  cout << "First letter of calculation type " << typeid(p).name() << endl;
  cout << "First letter of storage type " << typeid(mat[0]).name() << endl;
#endif
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / threads + 1;

  #pragma omp parallel num_threads(threads) firstprivate(x)
  { 
    int tid = omp_get_thread_num();
    long long my_start = start + tid * chunk_size;
    long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    C s;  //+1 or -1 
    C prod; //product of the elements in vector 'x'
    C my_p = 0;
    long long i = my_start;
    long long gray = (i-1) ^ ((i-1) >> 1);

    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
        for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
          x[rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        }
      }
    }

    prod = 1.0;
    int zero_num = 0;
    for (int j = 0; j < nov; j++) {
      if (x[j] == 0) {
        zero_num++;
      } else {
        prod *= x[j];  //product of the elements in vector 'x'
      }
    }
    int k;

    int prodSign = 1;
    if(i & 1LL) {
      prodSign = -1;
    }
    while (i < my_end) {
      //compute the gray code
      k = __builtin_ctzll(i);
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      
      for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
        if (x[rows[j]] == 0) {
          zero_num--;
          x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          prod *= x[rows[j]];  //product of the elements in vector 'x'
        } else {
          prod /= x[rows[j]];
          x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          if (x[rows[j]] == 0) {
            zero_num++;
          } else {
            prod *= x[rows[j]];  //product of the elements in vector 'x'
          }
        }
      }

      if(zero_num == 0) {
        my_p += prodSign * prod; 
      }
      prodSign *= -1;
      i++;
    }

    #pragma omp critical
      p += my_p;
  }
  
  
  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);

  //if(flags.type == "__float128"){
  //__float128 big_perman = (4*(nov&1)-2) * p;
  //printf("Difference: %.16e \n", big_perman - perman);
  //}
    
  return result;
}

//template <typename T> std::string type_name();

#ifdef AVX

struct matD
{
  double* _matvector;
  
  matD(){
    
    int succ = posix_memalign((void **) &_matvector, alignof(__m512d), 8*sizeof(double));

    if(succ != 0){
      std::cout << "Failed to allocate aligned memory in matD constructor.. " << std::endl;
      exit(1);
    }
  }
  
};

struct avxS
{
  __m512d vec;
};

void avx_x_debug(__m512d avx_x_part){
  
  double pr[8] __attribute__ ((aligned(64)));
  _mm512_store_pd(pr, avx_x_part);
  std::cout << "Part of x: ";
  std::cout << pr[0] << " ";
  std::cout << pr[1] << " ";
  std::cout << pr[2] << " ";
  std::cout << pr[3] << " ";
  std::cout << pr[4] << " ";
  std::cout << pr[5] << " ";
  std::cout << pr[6] << " ";
  std::cout << pr[7] << " ";
  
  printf("\n"); 
}

void avx_debug(double* mat_t, double** avx_copy_mat_t, __m512d* avx_mat, int nov, int avx_size, int avx_num){

  std::cout << std::fixed;
  std::cout << std::setprecision(1);
  
    
  printf("##########densemat##########\n");

  for(int i = 0; i < nov; i++){
    for(int j = 0; j < nov; j++){
      //printf("%.1e ", densemat->mat[i*nov+j]);
      std::cout << mat_t[i*nov+j] << " ";
    }
    printf("\n");
  }

  printf("##########densemat##########\n");


  printf("##########avx_copy_mat_t##########\n");
  for(int i = 0; i < nov; i++){
    for(int j = 0; j < avx_num; j++){
      std::cout << avx_copy_mat_t[i*avx_num+j][0] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][1] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][2] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][3] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][4] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][5] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][6] << " ";
      std::cout << avx_copy_mat_t[i*avx_num+j][7] << " ";
    }
    printf("\n");
  }
  printf("##########avx_copy_mat_t##########\n");


  
  printf("##########avx_mat(_m512d)##########\n");
  for(int i = 0; i < nov; i++){
    for(int j = 0; j < avx_num; j++){
      double pr[8] __attribute__ ((aligned(64)));
      _mm512_store_pd(pr, avx_mat[i*avx_num+j]);
      std::cout << pr[0] << " ";
      std::cout << pr[1] << " ";
      std::cout << pr[2] << " ";
      std::cout << pr[3] << " ";
      std::cout << pr[4] << " ";
      std::cout << pr[5] << " ";
      std::cout << pr[6] << " ";
      std::cout << pr[7] << " ";
    }
    printf("\n");
  }
  printf("##########avx_mat(_m512d)##########\n");  
}

template <class C, class S>
Result parallel_perman64_avx512(DenseMatrix<S>* densemat, flags flags) {

  //Pack parameters//
  S* mat = densemat->mat;
  int nov = densemat->nov;
  //Pack parameters//

  //Pack Flags//
  int threads = flags.threads;
  //Pack Flags//

  double starttime = omp_get_wtime();

  C x[nov];
  C rs; //row sum
  C p = 1; //Product of the elements in vector 'x'

  //create the x vector and initiate the permanent                                                  
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j                                                    
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry             
    p *= x[j];   // product of the elements in vector 'x'                                           
  }

  //create the transpose of the matrix                                                              
  S* mat_t = new S[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }


  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));

  long long chunk_size = end / threads + 1;

  int avx_num = (nov / 8) + 1; //How many avx vectors in a row
  //if(nov%8 == 0) avx_num -= 1;
  int avx_row = avx_num - 1; //How many avx vectors in a row which only include original values
  int original_values = avx_row * 8;
  int avx_size = avx_num * 8; //Total element count including 1 fillings
  int no_ones = avx_size - nov;
  int avx_row_size = nov + no_ones;
  
  int reverse_offset = 8 - no_ones;
  S** avx_copy_mat_t;
  
  int succ = posix_memalign((void **)&avx_copy_mat_t, alignof(__m512d), nov*avx_num*sizeof(double*));
  
  if(succ != 0)
    std::cout << "Could not succeed in posix_memalign at .. i: " << -1 << std::endl;
  
  for(int i = 0; i < nov * avx_num; i++){
    succ = posix_memalign((void **)&avx_copy_mat_t[i], alignof(__m512d), 8*sizeof(double)); //This is safe
    
    if(succ != 0){
      std::cout << "Could not succeed in posix_memalign at .. i: " << i << std::endl;
    }
  }
  
  for(int n = 0; n < nov; n++){
    for(int i = 0; i < avx_row; i++){
      avx_copy_mat_t[(n*avx_num)+i][0] = mat_t[n*nov + i*8 + 0];
      avx_copy_mat_t[(n*avx_num)+i][1] = mat_t[n*nov + i*8 + 1];
      avx_copy_mat_t[(n*avx_num)+i][2] = mat_t[n*nov + i*8 + 2];
      avx_copy_mat_t[(n*avx_num)+i][3] = mat_t[n*nov + i*8 + 3];
      avx_copy_mat_t[(n*avx_num)+i][4] = mat_t[n*nov + i*8 + 4];
      avx_copy_mat_t[(n*avx_num)+i][5] = mat_t[n*nov + i*8 + 5];
      avx_copy_mat_t[(n*avx_num)+i][6] = mat_t[n*nov + i*8 + 6];
      avx_copy_mat_t[(n*avx_num)+i][7] = mat_t[n*nov + i*8 + 7];
    }
    
    //Loop unrolling for "last" avx vector
    for(int i = 0; i < 8; i++){
      
      if(i < reverse_offset)
	avx_copy_mat_t[(n*avx_num)+avx_row][i] = mat_t[(n*nov) + original_values + i];
      else
	avx_copy_mat_t[(n*avx_num)+avx_row][i] = (double)1.0;
    } 
  } 

  
  __m512d* avx_mat;
  succ = posix_memalign((void **)&avx_mat, alignof(__m512d), nov*avx_num*sizeof(__m512d)); //This is safe
  
  
  
  for(int i = 0; i < nov; i++){
    for(int j = 0; j < avx_num; j++){
      avx_mat[(i*avx_num)+j] = _mm512_load_pd(avx_copy_mat_t[(i*avx_num)+j]);
    }
  }
  
  //avx_debug(mat_t, avx_copy_mat_t, avx_mat, nov, avx_size, avx_num);

  int to_mask = 0;

  for(int i = 0; i < reverse_offset; i++){
    to_mask += pow(2,i);
  }

  __mmask8 last_mask = _mm512_int2mask(to_mask);

        
#pragma omp parallel num_threads(threads) firstprivate(x)
  {
    int tid = omp_get_thread_num();
    long long my_start = start + tid * chunk_size;
    long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    C *xptr;
    
    
    C s;  //+1 or -1 
    C prod; //product of the elements in vector 'x'
    C my_p = 0;
    long long i = my_start;
    long long gray = (i-1) ^ ((i-1) >> 1);
    
    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
        xptr = (C*)x;
        for (int j = 0; j < nov; j++) {
          *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
          xptr++;
        }
      }
    }
      
    S** avx_x_copy;
    succ = posix_memalign((void **)&avx_x_copy, alignof(__m512d), avx_num*sizeof(double*));
    for(int i = 0; i < avx_num;i++){
      posix_memalign((void **) &avx_x_copy[i], alignof(__m512d), 8*sizeof(double));
    }
    
    for(int i = 0; i < nov; i++){
      avx_x_copy[i/8][i%8] = x[i];
    }
    
    for(int i = nov; i < avx_size; i++){
      avx_x_copy[avx_num-1][i%8] = (double)1.0;
    }
       
    __m512d* avx_x;// = new __m512d[avx_num];
    succ = posix_memalign((void **)&avx_x, alignof(__m512d), avx_num*sizeof(__m512d));
    
    for(int i = 0; i < avx_num; i++){
      //std::cout << "i: " << i << std::endl;
      avx_x[i] = _mm512_load_pd(avx_x_copy[i]); 
    }

    //std::cout << "###################" << std::endl;
    //for(int i = 0; i < avx_num; i++){
    //avx_x_debug(avx_x[i]);
    //}
 
    
    int prodSign = 1;
    if(i & 1LL) {
      prodSign = -1;
    }
    
    int k = 0;
    for (int i = my_start; i < my_end; i++) { //This should stay i
      
      //compute the gray code
      k = __builtin_ctzll(i);
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      prod = 1.0;
      //xptr = (C*)x;

      //We also have: add_ps -> for float, 32bit single precision
      //              add_ph -> for "half", 16 bit single precision, 32 elements at once
      
      if(s > 0){
	for (int j = 0; j < avx_num-1; j++) {
	  avx_x[j] = _mm512_add_pd(avx_x[j], avx_mat[(k*avx_num)+j]);
	}
	
	avx_x[avx_num-1] = _mm512_mask_add_pd(avx_x[avx_num-1], last_mask, avx_x[avx_num-1], avx_mat[k*avx_num+(avx_num-1)]);
      }
      
      else{
	for (int j = 0; j < avx_num-1; j++) {
	  avx_x[j] = _mm512_sub_pd(avx_x[j], avx_mat[(k*avx_num)+j]);
	}
	
	avx_x[avx_num-1] = _mm512_mask_sub_pd(avx_x[avx_num-1], last_mask, avx_x[avx_num-1], avx_mat[k*avx_num+(avx_num-1)]);
      }
      
      for(int j = 0; j < avx_num; j++){
	prod *= _mm512_reduce_mul_pd(avx_x[j]);
      }
            
      
      my_p += prodSign * prod; 
      prodSign *= -1;
      //i++;
    }
    
#pragma omp critical
    {
      p += my_p;
    }
  }

  delete [] mat_t;

  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);
  return result;
  
}

template <class C, class S>
Result parallel_perman64_avx512_32bit(DenseMatrix<S>* densemat, flags flags) {

  //Pack parameters//
  S* mat = densemat->mat;
  int nov = densemat->nov;
  //Pack parameters//
  
  //Pack Flags//
  int threads = flags.threads;
  //Pack Flags//
  
  double starttime = omp_get_wtime();
  
  C x[nov];
  C rs; //row sum
  C p = 1; //Product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent                                                  
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j                                                    
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry             
    p *= x[j];   // product of the elements in vector 'x'                                           
  }
  
  //create the transpose of the matrix                                                              
  S* mat_t = new S[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }
  
  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / threads + 1;
  
  int avx_num = (nov / 16) + 1; //How many avx vectors in a row
  //if(nov%8 == 0) avx_num -= 1;
  int avx_row = avx_num - 1; //How many avx vectors in a row which only include original values
  int original_values = avx_row * 16;
  int avx_size = avx_num * 16; //Total element count including 1 fillings
  int no_ones = avx_size - nov;
  int avx_row_size = nov + no_ones;
  
  int reverse_offset = 16 - no_ones;  
  S** avx_copy_mat_t;
  
  int succ = posix_memalign((void **)&avx_copy_mat_t, alignof(__m512), nov*avx_num*sizeof(double*));
  
  if(succ != 0)
    std::cout << "Could not succeed in posix_memalign at .. i: " << -1 << std::endl;
  
  for(int i = 0; i < nov * avx_num; i++){
    succ = posix_memalign((void **)&avx_copy_mat_t[i], alignof(__m512), 16*sizeof(double)); //This is safe
    
    if(succ != 0){
      std::cout << "Could not succeed in posix_memalign at .. i: " << i << std::endl;
    }
  }
  
  for(int n = 0; n < nov; n++){
    for(int i = 0; i < avx_row; i++){
      avx_copy_mat_t[(n*avx_num)+i][0] = mat_t[n*nov + i*16 + 0];
      avx_copy_mat_t[(n*avx_num)+i][1] = mat_t[n*nov + i*16 + 1];
      avx_copy_mat_t[(n*avx_num)+i][2] = mat_t[n*nov + i*16 + 2];
      avx_copy_mat_t[(n*avx_num)+i][3] = mat_t[n*nov + i*16 + 3];
      avx_copy_mat_t[(n*avx_num)+i][4] = mat_t[n*nov + i*16 + 4];
      avx_copy_mat_t[(n*avx_num)+i][5] = mat_t[n*nov + i*16 + 5];
      avx_copy_mat_t[(n*avx_num)+i][6] = mat_t[n*nov + i*16 + 6];
      avx_copy_mat_t[(n*avx_num)+i][7] = mat_t[n*nov + i*16 + 7];
      avx_copy_mat_t[(n*avx_num)+i][8] = mat_t[n*nov + i*16 + 8];
      avx_copy_mat_t[(n*avx_num)+i][9] = mat_t[n*nov + i*16 + 9];
      avx_copy_mat_t[(n*avx_num)+i][10] = mat_t[n*nov + i*16 + 10];
      avx_copy_mat_t[(n*avx_num)+i][11] = mat_t[n*nov + i*16 + 11];
      avx_copy_mat_t[(n*avx_num)+i][12] = mat_t[n*nov + i*16 + 12];
      avx_copy_mat_t[(n*avx_num)+i][13] = mat_t[n*nov + i*16 + 13];
      avx_copy_mat_t[(n*avx_num)+i][14] = mat_t[n*nov + i*16 + 14];
      avx_copy_mat_t[(n*avx_num)+i][15] = mat_t[n*nov + i*16 + 15];
    }

    
    //Loop unrolling for "last" avx vector
    for(int i = 0; i < 16; i++){
      
      if(i < reverse_offset)
	avx_copy_mat_t[(n*avx_num)+avx_row][i] = mat_t[(n*nov) + original_values + i];
      else
	avx_copy_mat_t[(n*avx_num)+avx_row][i] = (double)1.0;
    } 
  } 

  
  __m512* avx_mat;
  succ = posix_memalign((void **)&avx_mat, alignof(__m512), nov*avx_num*sizeof(__m512)); //This is safe
  
  
  
  for(int i = 0; i < nov; i++){
    for(int j = 0; j < avx_num; j++){
      avx_mat[(i*avx_num)+j] = _mm512_load_ps(avx_copy_mat_t[(i*avx_num)+j]);
    }
  }
  
  //avx_debug(mat_t, avx_copy_mat_t, avx_mat, nov, avx_size, avx_num);
  
  int to_mask = 0;

  for(int i = 0; i < reverse_offset; i++){
    to_mask += pow(2,i);
  }

  __mmask16 last_mask = _mm512_int2mask(to_mask);
    
    
#pragma omp parallel num_threads(threads) firstprivate(x)
  {
    int tid = omp_get_thread_num();
    long long my_start = start + tid * chunk_size;
    long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    C *xptr;
    
    C s;  //+1 or -1 
    C prod; //product of the elements in vector 'x'
    C my_p = 0;
    long long i = my_start;
    long long gray = (i-1) ^ ((i-1) >> 1);
    
    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
        xptr = (C*)x;
        for (int j = 0; j < nov; j++) {
          *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
          xptr++;
        }
      }
    }
    
    S** avx_x_copy;
    succ = posix_memalign((void **)&avx_x_copy, alignof(__m512), avx_num*sizeof(double*));
    for(int i = 0; i < avx_num;i++){
      posix_memalign((void **) &avx_x_copy[i], alignof(__m512), 16*sizeof(double));
    }
    
    for(int i = 0; i < nov; i++){
      avx_x_copy[i/16][i%16] = x[i];
    }
    
    for(int i = nov; i < avx_size; i++){
      avx_x_copy[avx_num-1][i%16] = (double)1.0;
    }
       
    __m512* avx_x;
    succ = posix_memalign((void **)&avx_x, alignof(__m512), avx_num*sizeof(__m512));
    
    for(int i = 0; i < avx_num; i++){
      avx_x[i] = _mm512_load_ps(avx_x_copy[i]);
    }

    //for(int i = 0; i < avx_num; i++){
    //avx_x_debug(avx_x[i]);
    //}
    
    int prodSign = 1;
    if(i & 1LL) {
      prodSign = -1;
    }
    
    int k = 0;
    for (int i = my_start; i < my_end; i++) { 
     
      //compute the gray code
      k = __builtin_ctzll(i);
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      prod = 1.0;
      
      //We also have: add_ps -> for float, 32bit single precision
      //              add_ph -> for "half", 16 bit single precision, 32 elements at once
      
      if(s > 0){
	for (int j = 0; j < avx_num-1; j++) {
	  avx_x[j] = _mm512_add_ps(avx_x[j], avx_mat[(k*avx_num)+j]);
	}
	avx_x[avx_num-1] = _mm512_mask_add_ps(avx_x[avx_num-1], last_mask, avx_x[avx_num-1], avx_mat[k*avx_num+(avx_num-1)]);
      }
      
      else{
	for (int j = 0; j < avx_num-1; j++) {
	  avx_x[j] = _mm512_sub_ps(avx_x[j], avx_mat[(k*avx_num)+j]);
	}
	avx_x[avx_num-1] = _mm512_mask_sub_ps(avx_x[avx_num-1], last_mask, avx_x[avx_num-1], avx_mat[k*avx_num+(avx_num-1)]);
      }

      for(int j = 0; j < avx_num; j++){
	prod *= _mm512_reduce_mul_ps(avx_x[j]);
      }
      
      my_p += prodSign * prod; 
      prodSign *= -1;
      //i++;
    }
    
#pragma omp critical
    {
      p += my_p;
    }
  }

  delete [] mat_t;

  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);
  return result;
}

#endif

template <class C, class S>
Result parallel_perman64(DenseMatrix<S>* densemat, flags flags) {

  std::cout << std::fixed;
  std::cout << std::setprecision(1);
  
  //Pack parameters//
  S* mat = densemat->mat;
  int nov = densemat->nov;
  //Pack parameters//
  
  //Pack flags//
  int threads = flags.threads;
  //Pack flags//

  double starttime = omp_get_wtime();
  
  C x[nov];   
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'
  
  //cout << "First letter of calculation type " << typeid(p).name() << endl;
  //cout << "First letter of storage type " << typeid(mat[0]).name() << endl;

  //std::string ctype(typeid(p).name());
  //std::string stype(typeid(mat[0]).name());

  
  //std::cout << "ctype: " << type_name<decltype(p)>() << '\n';
  //std::cout << "stype: " << type_name<decltype(mat[0])>() << '\n';
  
    
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  S* mat_t = new S[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / threads + 1;

  #pragma omp parallel num_threads(threads) firstprivate(x)
  { 
    int tid = omp_get_thread_num();
    long long my_start = start + tid * chunk_size;
    long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    C *xptr; 
    C s;  //+1 or -1 
    C prod; //product of the elements in vector 'x'
    C my_p = 0;
    long long i = my_start;
    long long gray = (i-1) ^ ((i-1) >> 1);
    
    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
        xptr = (C*)x;
        for (int j = 0; j < nov; j++) {
          *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
          xptr++;
        }
      }
    }
    
    int prodSign = 1;
    int k = 0;
    if(i & 1LL) {
      prodSign = -1;
    }
    
    while (i < my_end) {
      //compute the gray code
      k = __builtin_ctzll(i);
      if(i < 100)std:cout << "i: " << i << " k: " << k << std::endl;
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      
      prod = 1.0;
      xptr = (C*)x;
      for (int j = 0; j < nov; j++) {
        *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= *xptr++;  //product of the elements in vector 'x'
      }

      /*
      //x debug
      xptr = (C*)x;
      std::cout << "################" << std::endl;
      std::cout << "i: " << i << std::endl;
      for(int m = 0; m < 4; m++){
	for(int l = 0; l < 8; l++){
	  std::cout << (double)xptr[m*8+l] << " ";
	}
	std::cout << std::endl;
      }
      std::cout << std::endl;
      //x debug
      */

      my_p += prodSign * prod; 
      prodSign *= -1;
      i++;
    }

#pragma omp critical
    {
      p += my_p;
    }
  }

  delete [] mat_t;

  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);
  return result;
}

template <class C, class S>
Result parallel_skip_perman64_w(SparseMatrix<S>* sparsemat, flags flags) {

  //Pack parameters//
  int* rptrs = sparsemat->rptrs;
  int* cols = sparsemat->cols;
  S* rvals = sparsemat->rvals;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters//
  
  //Pack flags//
  int threads = flags.threads;
  //Pack flags//

  //first initialize the vector then we will copy it to ourselves
  C rs, x[64], p;
  int j, ptr;
  unsigned long long ci, start, end, chunk_size, change_j;
  
  double starttime = omp_get_wtime();

  //initialize the vector entries                                                                                        
  for (j = 0; j < nov; j++) {
    rs = .0f; 
    for(ptr = rptrs[j]; ptr < rptrs[j+1]; ptr++) 
      rs += rvals[ptr];
    x[j] = -rs/(2.0f);
  }
  
  for(ptr = cptrs[nov-1]; ptr < cptrs[nov]; ptr++) {
    x[rows[ptr]] += cvals[ptr];
  }

  //update perman with initial x
  C prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j];
  }
  p = prod;

  //find start location        
  start = 1;
  for(int j = 0; j < nov; j++) {
    if(x[j] == 0) {
      change_j = -1;
      for (ptr = rptrs[j]; ptr < rptrs[j + 1]; ptr++) {
        ci = 1ULL << cols[ptr]; 
        if(ci < change_j) {
          change_j = ci;
        }
      }
      if(change_j > start) {
        start = change_j;
      }
    }
  }

  end = (1ULL << (nov-1));
  
  chunk_size = (end - start + 1) / threads + 1;

  #pragma omp parallel num_threads(threads) private(j, ci, change_j) 
  {
    C my_x[64];
    memcpy(my_x, x, sizeof(C) * 64);
    
    int tid = omp_get_thread_num();
    unsigned long long my_start = start + tid * chunk_size;
    unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    //update if neccessary
    C my_p = 0;

    unsigned long long my_gray;    
    unsigned long long my_prev_gray = 0;

    int ptr, last_zero;
    unsigned long long period, steps, step_start;

    unsigned long long i = my_start;

    while (i < my_end) {
      //k = __builtin_ctzll(i + 1);
      my_gray = i ^ (i >> 1);

      unsigned long long gray_diff = my_prev_gray ^ my_gray;
      //cout << "X: " << gray_diff << endl;
      j = 0;
      while(gray_diff > 0) { // this contains the bit to be updated
        unsigned long long onej = 1ULL << j;
        if(gray_diff & onej) { // if bit l is changed 
          gray_diff ^= onej;   // unset bit
          if(my_gray & onej) {    // do the update
            for (ptr = cptrs[j]; ptr < cptrs[j + 1]; ptr++) {
              my_x[rows[ptr]] += cvals[ptr];
            }
          }
          else {
            for (ptr = cptrs[j]; ptr < cptrs[j + 1]; ptr++) {
              my_x[rows[ptr]] -= cvals[ptr];
            }
          }
        }
        j++;
      }
      //counter++;
      my_prev_gray = my_gray;
      last_zero = -1;
      C my_prod = 1; 
      for(j = nov - 1; j >= 0; j--) {
        my_prod *= my_x[j];
        if(my_x[j] == 0) {
          last_zero = j;
          break;
        }
      }
      
      if(my_prod != 0) {
        my_p += ((i&1ULL)? -1.0:1.0) * my_prod;
        i++;
      }
      else {
        change_j = -1;
        for (ptr = rptrs[last_zero]; ptr < rptrs[last_zero + 1]; ptr++) {
          step_start = 1ULL << cols[ptr]; 
          period = step_start << 1; 
          ci = step_start;
          if(i >= step_start) {
            steps = (i - step_start) / period;
            ci = step_start + ((steps + 1) * period);
          }
          if(ci < change_j) {
            change_j = ci;
          }
        }
  
        i++;
        if(change_j > i) {
          i = change_j;
        } 
      }
    }

#pragma omp critical
    {
      p += my_p;
      //printf("tid is: %d -- p is: %f\n", tid, p);
    }
  }

  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);
  return result;
}


template <class C, class S>
Result parallel_skip_perman64_w_balanced(SparseMatrix<S>* sparsemat, flags flags) {

  //Pack parameters//
  int* rptrs = sparsemat->rptrs;
  int* cols = sparsemat->cols;
  S* rvals = sparsemat->rvals;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters//
  
  //Pack flags//
  int threads = flags.threads;
  //Pack flags//

  double starttime = omp_get_wtime();
  
  //first initialize the vector then we will copy it to ourselves
  C rs, x[nov], p;
  int j, ptr;
  unsigned long long ci, start, end, chunk_size, change_j;

#ifdef DTYPE
  cout << "First letter of calculation type " << typeid(p).name() << endl;
  cout << "First letter of storage type " << typeid(rvals[0]).name() << endl;
#endif
  
  //initialize the vector entries                                                                                        
  for (j = 0; j < nov; j++) {
    rs = .0f; 
    for(ptr = rptrs[j]; ptr < rptrs[j+1]; ptr++) 
      rs += rvals[ptr];
    x[j] = -rs/(2.0f);
  }
  
  for(ptr = cptrs[nov-1]; ptr < cptrs[nov]; ptr++) {
    x[rows[ptr]] += cvals[ptr];
  }

  //for(j = 0; j < nov; j++) {
  //cout << ((double)x[j]) << " ";
  //}
  //cout << endl;
  
  //update perman with initial x
  C prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j];
  }
  p = prod;

  //find start location        
  start = 1;
  for(int j = 0; j < nov; j++) {
    if(x[j] == 0) {
      change_j = -1;
      for (ptr = rptrs[j]; ptr < rptrs[j + 1]; ptr++) {
        ci = 1ULL << cols[ptr]; 
        if(ci < change_j) {
          change_j = ci;
        }
      }
      if(change_j > start) {
        start = change_j;
      }
    }
  }

  end = (1ULL << (nov-1));

  int no_chunks = 512;
  //int no_chunks = 2048;
  chunk_size = (end - start + 1) / no_chunks + 1;
  
#pragma omp parallel num_threads(threads) private(j, ci, change_j) 
  {
    C my_x[nov];
    
#pragma omp for schedule(dynamic, 1)
    for(int cid = 0; cid < no_chunks; cid++) {
      //    int tid = omp_get_thread_num();
      unsigned long long my_start = start + cid * chunk_size;
      unsigned long long my_end = min(start + ((cid+1) * chunk_size), end);
      
      //update if neccessary
      C my_p = 0;
      
      unsigned long long my_gray;    
      unsigned long long my_prev_gray = 0;
      memcpy(my_x, x, sizeof(C) * nov);
      
      int ptr, last_zero;
      unsigned long long period, steps, step_start;
      
      unsigned long long i = my_start;
      
      while (i < my_end) {
	//k = __builtin_ctzll(i + 1);
	my_gray = i ^ (i >> 1);
        
	unsigned long long gray_diff = my_prev_gray ^ my_gray;
        
	j = 0;
	while(gray_diff > 0) { // this contains the bit to be updated
	  unsigned long long onej = 1ULL << j;
	  if(gray_diff & onej) { // if bit l is changed 
	    gray_diff ^= onej;   // unset bit
	    if(my_gray & onej) {    // do the update
	      for (ptr = cptrs[j]; ptr < cptrs[j + 1]; ptr++) {
		my_x[rows[ptr]] += cvals[ptr];
	      }
	    }
	    else {
	      for (ptr = cptrs[j]; ptr < cptrs[j + 1]; ptr++) {
		my_x[rows[ptr]] -= cvals[ptr];
	      }
	    }
	  }
	  j++;
	}
	//counter++;
	my_prev_gray = my_gray;
	last_zero = -1;
	
	C my_prod = 1;
	
	for(j = nov - 1; j >= 0; j--) {
	  my_prod *= my_x[j];
	  if(my_x[j] == 0) {
	    last_zero = j;
	    break;
	  }
	}
	
	
	if(my_prod != 0) {
	  my_p += ((i&1ULL)? -1.0:1.0) * my_prod;
	  i++;
	} 
	else {
	  change_j = -1;
	  for (ptr = rptrs[last_zero]; ptr < rptrs[last_zero + 1]; ptr++) {
	    step_start = 1ULL << cols[ptr]; 
	    period = step_start << 1; 
	    ci = step_start;
	    if(i >= step_start) {
	      steps = (i - step_start) / period;
	      ci = step_start + ((steps + 1) * period);
	    }
	    if(ci < change_j) {
	      change_j = ci;
	    }
	  }
	  
	  i++;
	  if(change_j > i) {
	    i = change_j;
	  } 
	}
      }
      
#pragma omp critical
      p += my_p;
    }
  }
  
  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);
  
  //if(flags.type == "__float128"){
  //__float128 big_perman = (4*(nov&1)-2) * p;
  //printf("Difference: %.16e \n", big_perman - perman);
  //}
  
  return result;
}







template <class C, class S>
Result perman64(S* mat, int nov) {
  C x[64];   
  C rs; //row sum
  C s;  //+1 or -1 
  C prod; //product of the elements in vector 'x'
  C p = 1; //product of the elements in vector 'x'
  C *xptr; 
  int j, k;
  unsigned long long i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;

  double starttime = omp_get_wtime();
  
  //create the x vector and initiate the permanent
  for (j = 0; j < nov; j++) {
    rs = .0f;
    for (k = 0; k < nov; k++)
      rs += mat[(j * nov) + k];  // sum of row j
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  S* mat_t = new S[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  gray = 0;
  unsigned long long one = 1;

  unsigned long long counter = 0;

  double t_start = omp_get_wtime();
  for (i = 1; i <= tn11; i++) {

    //compute the gray code
    k = __builtin_ctzll(i);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
    
    counter++;
    prod = 1.0;
    xptr = (C*)x;
    for (j = 0; j < nov; j++) {
      *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= *xptr++;  //product of the elements in vector 'x'
    }

    p += ((i&1ULL)? -1.0:1.0) * prod; 
  }

  delete [] mat_t;

  double duration = omp_get_wtime() - starttime;  
  double perman = (4*(nov&1)-2) * p;
  Result result(perman, duration);
  return result;
}


template <class T>
double brute_w(int *xadj, int *adj, T* val, int nov) {
  double perman = 0;
  double prod;

  int* matched = new int[nov];
  for(int i = 0; i < nov; i++) matched[i] = 0;
  
  int h_nov = nov/2;
  int* ptrs = new int[h_nov];
  for(int i = 0; i < h_nov; i++) {ptrs[i] = xadj[i];}

  matched[0] = adj[0];
  matched[adj[0]] = 1;
  ptrs[0] = 1;
  prod = val[0];

  int curr = 1;
  while(curr >= 0) {
    //clear the existing matching on current
    if(matched[curr] != 0) {
      prod /= val[ptrs[curr] - 1];
      matched[matched[curr]] = 0;
      matched[curr] = 0;
    }

    //check if we can increase the matching by matching curr
    int ptr = ptrs[curr];
    int partner;
    for(; ptr < xadj[curr + 1]; ptr++) {
      if(matched[adj[ptr]] == 0) {
  partner = adj[ptr];
  ptrs[curr] = ptr + 1;
  prod *= val[ptr];
  break;
      }
    }

    if(ptr < xadj[curr + 1]) { //we can extend matching
      if(curr == h_nov - 1) {
  perman += prod;
  prod /= val[ptr];   
  ptrs[curr] = xadj[curr];
  curr--;
      } else {
  matched[curr] = partner;
  matched[partner] = 1;
  curr++;
      }
    } else {
      ptrs[curr] = xadj[curr];
      curr--;
    }
  }
  return perman;
}




#endif
