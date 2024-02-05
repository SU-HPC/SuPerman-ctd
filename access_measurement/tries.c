#include <iostream>
#include <stdio.h>


long long gcs(long long remaining, int grid_dim, int block_dim){

  long long chunk_size = 1;
  int no_threads = grid_dim * block_dim * 32;

  while((chunk_size * no_threads) < remaining){
    chunk_size *= 2;
  }

  chunk_size /= 2;
  return chunk_size;
}


int main(){

  long long a = 1LL << 32;
  std::cout << "a: " << a << std::endl;


  //for(long long i = 1; i < a; i++){
  //int k = __builtin_ctzll(i);
  //std::cout << "i: " << i << " k: " << k << std::endl;
  //}

  int grid_dim = 160;
  int block_dim = 384;

  int no_threads = grid_dim * block_dim * 32;
  long long chunk_size = 1;

  //while((chunk_size * no_threads) < a){
  //chunk_size *= 2;
  //std::cout << "Chunk size: " << chunk_size << std::endl;
  //}
  //chunk_size /= 2;

  //long long remaining =  a - (no_threads * chunk_size);
  
  //std::cout << "a: " << a << " chunk_size: " << chunk_size << " remaining: " << remaining << std::endl;

  //std::cout << "remaining / space: " << (double)(remaining) / (double)(a) << std::endl;

  //long long new_chunk_size = gcs(remaining, grid_dim, block_dim);
  //std::cout << "new chunk size: " << new_chunk_size << std::endl;
  //std::cout << "remaining / space: " << (double)(remaining - (new_chunk_size * no_threads)) / (double)(a) << std::endl;
  //std::cout << "new remaining: " << remaining - (new_chunk_size * no_threads) << std::endl;


  long long remaining = a;
  for(int i = 0; i < 16; i++){
    chunk_size = gcs(remaining, grid_dim, block_dim);
    remaining -= (no_threads * chunk_size);
    std::cout << "chunk_size: " << chunk_size << " completed: " << 1 - ((double)remaining / a) << " remaining: " << remaining << std::endl;
  }
  
  
  return 0;

}
