#include<stdio.h>
#include<stdlib.h>
#define BLOCK_DIM 32


typedef struct matriz{
    int x;
    int y;
    float *vetor;
    float *cuda_vetor;
    int num_de_pesos;


} TMATRIZ;



TMATRIZ  ler_matriz(char* nome_arq, int eh_entrada){
    FILE *f = fopen(nome_arq,"r");

    TMATRIZ my_matriz;
    if (eh_entrada){
        int x,y,num_de_pesos;
        fscanf(f,"%d %d %d\n",&x,&y,&num_de_pesos );
        my_matriz.x = x;
        my_matriz.y = y;
        my_matriz.num_de_pesos = num_de_pesos;
    }
    else{

         int x,y;
        fscanf(f,"%d %d\n",&x,&y );
        
        my_matriz.x = x;
        my_matriz.y = y;
        
    }

    my_matriz.vetor = (float*) malloc(sizeof(float)* my_matriz.x * my_matriz.y);
    
    for (int i = 0 ; i <  my_matriz.x * my_matriz.y; i++ )
        fscanf(f, "%f",  &my_matriz.vetor[i]);
    fclose(f);
    cudaMalloc(&(my_matriz.cuda_vetor), sizeof(float)* my_matriz.x * my_matriz.y );
    cudaMemcpy(my_matriz.cuda_vetor,my_matriz.vetor,sizeof(float) * my_matriz.x * my_matriz.y, cudaMemcpyHostToDevice);
    

    return my_matriz;
}

#include <math.h>

__device__
float sigmoid(float x) {
     return 1 / (1 + exp(-x));
}





__global__
void mm_kernel(float *mat1, float *mat2, float *mat3, int row1,int col1,int row2,int col2){  
    
    __shared__ float mat1_local[BLOCK_DIM][BLOCK_DIM];
    __shared__ float mat2_local[BLOCK_DIM][BLOCK_DIM];
    float soma = 0;


    for( int local_idx_block = 0; local_idx_block < ceilf((float)col1/BLOCK_DIM);local_idx_block++   ){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = local_idx_block * blockDim.y + threadIdx.y;
         
            
        if ((i <  row1 ) && (j < col1)){
            mat1_local[threadIdx.x][threadIdx.y] = mat1[i * col1 + j ];
        }
        else{
            mat1_local[threadIdx.x][threadIdx.y] = 0 ;
        }

        i = local_idx_block * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;
        if ((i <  row2 ) && (j < col2)){
            mat2_local[threadIdx.x][threadIdx.y] = mat2[i * col2 + j ];
        }
        else{
            mat2_local[threadIdx.x][threadIdx.y] = 0 ;
        }
        __syncthreads();
        for(int k = 0; k < BLOCK_DIM; k++){

            
           soma+= mat1_local[threadIdx.x][k] * mat2_local[k][threadIdx.y];
        }
        
         
        __syncthreads();
        
         

    }
    
         
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < row1) && (j < col1))    {
        mat3[i *  col2+ j] = sigmoid(soma);
    }
        
   

}





int main(){
    TMATRIZ entrada = ler_matriz("input/entrada", 1);
    char file_name[] = "input/peso-0";
    
    
    TMATRIZ *pesos = (TMATRIZ*) malloc(sizeof(TMATRIZ) * entrada.num_de_pesos);
    for(int i = 0; i < entrada.num_de_pesos; i++){
        pesos[i] = ler_matriz(file_name, 0);
        file_name[11]+=1;
    }
    float * matriz_atual = entrada.cuda_vetor;
    int x_atual = entrada.x;
    int y_atual = entrada.y;

    dim3 threads_per_block(32, 32);
    dim3 blocks_per_grid(  1, 1);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0 ; i < entrada.num_de_pesos;i++){
        float *temp;
        cudaMalloc(&temp, sizeof(float) * x_atual * pesos[i].y);
        blocks_per_grid.x =  std::ceil(  (float)x_atual / 32);
        blocks_per_grid.y = std::ceil( (float)pesos[i].y / 32);
        mm_kernel<<<blocks_per_grid,threads_per_block>>>(matriz_atual, pesos[i].cuda_vetor, temp,x_atual, y_atual,y_atual, pesos[i].y);
        cudaDeviceSynchronize();
        cudaFree(matriz_atual);
        matriz_atual = temp;
        cudaFree(pesos[i].cuda_vetor); 
        y_atual = pesos[i].y;
    }
    cudaEventRecord(stop);
    float milisegundos;
    cudaEventElapsedTime(&milisegundos, start,stop);
    FILE *f = fopen( "gpu_otimizado.output" ,"a");
    fprintf(f, "%f\n", milisegundos);
    fclose(f);
    

    cudaFree(matriz_atual);
    





    
    
}