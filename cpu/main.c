#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include <time.h>


typedef struct matriz{
    int x;
    int y;
    float *vetor;
    float *vetor_cuda;
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
    

    return my_matriz;
}

#include <math.h>

float sigmoid(float x) {
     return 1 / (1 + exp(-x));
}

float * multiplica_matriz(float *mat1, float* mat2, int x, int y ,int w){
    
    #pragma omp parallel for
    float *mat3 = (float*) malloc(sizeof(float)*x*w);
    for(int i = 0 ; i < x; i++){
        for (int j =0; j < w; j++){
            float temp=0;
            for(int k = 0; k < y; k++){
                temp+=mat1[i*y + k]* mat2[k*w+j];
            }
            mat3[i*w+j] = sigmoid(temp);
            
        }
    }
    
    return mat3;
}




int main(){
    TMATRIZ entrada = ler_matriz("input/entrada", 1);
    char file_name[] = "input/peso-0";

    TMATRIZ *pesos = (TMATRIZ*) malloc(sizeof(TMATRIZ) * entrada.num_de_pesos);
    for(int i = 0; i < entrada.num_de_pesos; i++){
        pesos[i] = ler_matriz(file_name, 0);
        file_name[11]+=1;

    }
    float * matriz_atual = entrada.vetor;
    int x_atual = entrada.x;
    int y_atual = entrada.y;

    


    clock_t start = clock();
    for (int i = 0 ; i < entrada.num_de_pesos;i++){
        float *temp_matriz = multiplica_matriz(matriz_atual, pesos[i].vetor, x_atual, y_atual, pesos[i].y);
        free(matriz_atual);
        matriz_atual = temp_matriz;
        
        y_atual = pesos[i].y;
    }
    clock_t end = clock();
    double elapsed = (double)(end - start)/CLOCKS_PER_SEC;
    FILE *f = fopen( "output_cpu_normal.output" ,"a");
    fprintf(f, "%f\n", elapsed);
    fclose(f);
    printf("%f\n", matriz_atual[0]);
    // printf("Time measured: %f seconds.\n", elapsed);

}