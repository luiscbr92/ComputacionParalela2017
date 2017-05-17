/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código secuencial base
*
* @author Ana Moretón Fernández, Arturo Gonzalez-Escribano
* @version v1.3
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include "cputils.h"

/* Substituir min por el operador */
#define min(x,y)    ((x) < (y)? (x) : (y))

// Definición de constantes
#define currentGPU 0
#define BLOCK_DIM_FILAS 128
#define BLOCK_DIM_COLUMNAS 8
#define MAX_THREADS_PER_BLOCK 1024

__global__ void etiquetadoInicialKernel(int* matrixDataDev, int* matrixResultDev, int* matrixResultCopyDev,int rows, int columns){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i != 0 && j != 0 && i < rows-1 && j < columns-1){
		matrixResultCopyDev[i*(columns)+j]=-1;
		matrixResultDev[i*(columns)+j]=-1;
		// Si es 0 se trata del fondo y no lo computamos
		if(matrixDataDev[i*(columns)+j]!=0)
			matrixResultDev[i*(columns)+j]=i*(columns)+j;
	}
}

__global__ void inicializaMatrixFlag(int* matrixFlagDev, int rows, int columns){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < rows && j < columns)
		matrixFlagDev[i*columns+j] = 0;
}

__global__ void actualizacionCopiaKernel(int *matrixResultDev, int *matrixResultCopyDev, int rows, int columns){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i != 0 && j != 0 && i < rows-1 && j < columns-1){
		matrixResultCopyDev[i*(columns)+j]=matrixResultDev[i*(columns)+j];
	}
}

__global__ void computationKernel(int *matrixDataDev, int *matrixResultDev, int *matrixResultCopyDev, int *matrixFlagDev, int rows, int columns){
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i != 0 && j != 0 && i < rows-1 && j < columns-1){
		// Inicialmente cojo mi indice
		int result=matrixResultCopyDev[i*columns+j];
		if(result == -1)
			matrixFlagDev[i*columns+j] = 0;
		else{
			//Si es de mi mismo grupo, entonces actualizo
			if(matrixDataDev[(i-1)*columns+j] == matrixDataDev[i*columns+j])
			{
				result = min (result, matrixResultCopyDev[(i-1)*columns+j]);
			}
			if(matrixDataDev[(i+1)*columns+j] == matrixDataDev[i*columns+j])
			{
				result = min (result, matrixResultCopyDev[(i+1)*columns+j]);
			}
			if(matrixDataDev[i*columns+j-1] == matrixDataDev[i*columns+j])
			{
				result = min (result, matrixResultCopyDev[i*columns+j-1]);
			}
			if(matrixDataDev[i*columns+j+1] == matrixDataDev[i*columns+j])
			{
				result = min (result, matrixResultCopyDev[i*columns+j+1]);
			}

			// Si el indice no ha cambiado retorna 0
			if(matrixResultDev[i*columns+j] == result)
				matrixFlagDev[i*columns+j] = 0;
			// Si el indice cambia, actualizo matrix de resultados con el indice adecuado y retorno 1
			else {
				matrixResultDev[i*columns+j]=result;
				matrixFlagDev[i*columns+j] = 1;
			}
		}
	}
}


__global__ void reduce_kernel(const int* g_idata, int* g_odata){
    extern __shared__ int sdata[];

    // cada hilo carga un elemento desde memoria global hacia memoria shared
    unsigned int tid = threadIdx.x;
    unsigned int igl = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[igl];
    __syncthreads();

    // Hacemos la reducción en memoria shared
    for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
	    // Comprobamos si el hilo actual es activo para esta iteración
      if (tid < s){
	       // Hacemos la reducción sumando los dos elementos que le tocan a este hilo
          sdata[tid] += sdata[tid+s];
	    }
	    __syncthreads();
    }

    // El hilo 0 de cada bloque escribe el resultado final de la reducción
    // en la memoria global del dispositivo pasada por parámetro (g_odata[])
    if (tid == 0)
      g_odata[blockIdx.x] = sdata[0];

}



int reduce(int* values, unsigned int numValues){
	// ME PREOCUPA QUE EL numValues NO SEA CORRECTO.


	cudaError_t errCuda;
	// Si el tamaño del vector es impar, sumo el ultimo elemento al primero
	if(numValues % 2 == 1)
		values[0] += values[numValues-1];

	// Para almacenar el resultado final
	int *result = NULL;
	result= (int *)malloc(sizeof(int) );
	if ( (result == NULL)   ) {
 		perror ("Error reservando memoria");
	}

	int numThreadsPerBlock = MAX_THREADS_PER_BLOCK;
	int numBlocks;
	if(numValues % (numThreadsPerBlock*2) == 0)
		numBlocks = numValues / (numThreadsPerBlock*2);
	else
		numBlocks = numValues / (numThreadsPerBlock*2) +1;

	int *d_Result = NULL;
	errCuda = cudaMalloc(&d_Result, numBlocks* sizeof(int));
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se inicializo d_Result. Saliendo...\n");
	}
	int sharedMemorySize = numThreadsPerBlock*2 * sizeof(int);
	//La primera pasada reduce el array de entrada: VALUES
  //a un array de igual tamaño que el número total de bloques del grid: D_RESULT
	reduce_kernel<<<numBlocks,numThreadsPerBlock,sharedMemorySize>>>(values,d_Result);
	errCuda = cudaGetLastError();
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("Fallo primera fase de reduccion. Saliendo...\n");
	}

	while(numBlocks > MAX_THREADS_PER_BLOCK){
		if(numBlocks % (numThreadsPerBlock*2) == 0)
			numBlocks = numBlocks / (numThreadsPerBlock*2);
		else
			numBlocks = numBlocks / (numThreadsPerBlock*2) +1;

		reduce_kernel<<<numBlocks,numThreadsPerBlock,sharedMemorySize>>>(d_Result,d_Result);
		errCuda = cudaGetLastError();
		if(errCuda != cudaSuccess){
			printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
			printf("Fallo primera fase de reduccion. Saliendo...\n");
		}
	}

  //La segunda pasada lanza sólo un único bloque para realizar la reducción final
  numThreadsPerBlock = numBlocks;
  numBlocks = 1;
  sharedMemorySize = numThreadsPerBlock * sizeof(int);
	// printf("Lanzando segundo reduce %d %d %d\n", numBlocks, numThreadsPerBlock, sharedMemorySize);
  reduce_kernel<<<numBlocks, numThreadsPerBlock, sharedMemorySize>>>(d_Result, d_Result);
	errCuda = cudaGetLastError();
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("Fallo segunda fase de reduccion. Saliendo...\n");
	}

	errCuda =cudaMemcpy(result, d_Result, sizeof(int), cudaMemcpyDeviceToHost);
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("error haciendo una Transferencia. Saliendo...\n");
	}

	//printf("%d\n", result[0]);
	int value = result[0];
  return value;
}

/**
* Funcion principal
*/
int main (int argc, char* argv[])
{

	/* 1. Leer argumento y declaraciones */
	if (argc < 2) 	{
		printf("Uso: %s <imagen_a_procesar>\n", argv[0]);
		return(EXIT_SUCCESS);
	}
	char* image_filename = argv[1];

	int rows=-1;
	int columns =-1;
	int *matrixData=NULL, *matrixDataDev=NULL;
	int *matrixResult=NULL, *matrixResultDev=NULL;
	int *matrixResultCopy=NULL, *matrixResultCopyDev=NULL;
	int numBlocks=-1;



	/* 2. Leer Fichero de entrada e inicializar datos */

	/* 2.1 Abrir fichero */
	FILE *f = cp_abrir_fichero(image_filename);

	// Compruebo que no ha habido errores
	if (f==NULL)
	{
	   perror ("Error al abrir fichero.txt");
	   return -1;
	}

	/* 2.2 Leo valores del fichero */
	int i,j;
	fscanf (f, "%d\n", &rows);
	fscanf (f, "%d\n", &columns);
	// Añado dos filas y dos columnas mas para los bordes
	rows=rows+2;
	columns = columns+2;

	/* 2.3 Reservo la memoria necesaria para la matriz de datos */
	matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
	if ( (matrixData == NULL)   ) {
 		perror ("Error reservando memoria");
	   	return -1;
	}

	/* 2.4 Inicializo matrices */
	for(i=0;i< rows; i++){
		for(j=0;j< columns; j++){
			matrixData[i*(columns)+j]=-1;
		}
	}
	/* 2.5 Relleno bordes de la matriz */
	for(i=1;i<rows-1;i++){
		matrixData[i*(columns)+0]=0;
		matrixData[i*(columns)+columns-1]=0;
	}
	for(i=1;i<columns-1;i++){
		matrixData[0*(columns)+i]=0;
		matrixData[(rows-1)*(columns)+i]=0;
	}
	/* 2.6 Relleno la matriz con los datos del fichero */
	for(i=1;i<rows-1;i++){
		for(j=1;j<columns-1;j++){
			fscanf (f, "%d\n", &matrixData[i*(columns)+j]);
		}
	}
	fclose(f);

	#ifdef WRITE
		printf("Inicializacion \n");
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixData[i*(columns)+j]);
			}
			printf("\n");
		}
	#endif

	cudaSetDevice(0);
	cudaDeviceSynchronize();

	/* PUNTO DE INICIO MEDIDA DE TIEMPO */
	double t_ini = cp_Wtime();

//
// EL CODIGO A PARALELIZAR COMIENZA AQUI
//

	// Calculo de grid y bloques general
	dim3 block(BLOCK_DIM_COLUMNAS,BLOCK_DIM_FILAS);
	int num_col_grid, num_fil_grid;
	if(columns % BLOCK_DIM_COLUMNAS != 0)
		num_col_grid = columns/BLOCK_DIM_COLUMNAS+1;
	else
		num_col_grid = columns/BLOCK_DIM_COLUMNAS;
	if(rows % BLOCK_DIM_FILAS != 0)
		num_fil_grid = rows/BLOCK_DIM_FILAS+1;
	else
		num_fil_grid = rows/BLOCK_DIM_FILAS;

	//printf("F%d-C%d\n", num_fil_grid, num_col_grid);
	dim3 grid(num_col_grid, num_fil_grid);

	//numBlocks = num_col_grid * num_fil_grid;
	//printf ("%d\n",numBlocks);

	// Control de errores
	cudaError_t errCuda;

	//Iniclializacion de estructuras
	errCuda = cudaMalloc(&matrixDataDev, rows*columns* sizeof(int));
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se inicializo matrixDataDev. Saliendo...\n");
		return -1;
	}

  matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int));
  matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
  if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
     perror ("Error reservando memoria");
       return -1;
  }

	errCuda = cudaMalloc(&matrixResultDev, rows*columns* sizeof(int));
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se inicializo matrixResultDev. Saliendo...\n");
		return -1;
	}

	errCuda = cudaMalloc(&matrixResultCopyDev, rows*columns* sizeof(int));
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se inicializo matrixResultCopyDev. Saliendo...\n");
		return -1;
	}

	//Transferencia de matrixData a device
  errCuda = cudaMemcpy(matrixDataDev, matrixData, rows*columns* sizeof(int), cudaMemcpyHostToDevice);
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se copio matrixData a matrixDataDev. Saliendo...\n");
		return -1;
	}

	// Inicialización de la matriz que controla los flags
	int *matrixFlag=NULL, *matrixFlagDev=NULL;
	matrixFlag= (int *)malloc( (rows)*(columns) * sizeof(int) );
  if ( (matrixFlag == NULL) ) {
    perror ("Error reservando memoria");
    return -1;
  }

	errCuda = cudaMalloc(&matrixFlagDev, rows*columns* sizeof(int));
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se reservo bien el espacio para matrixFlagDev. Saliendo...\n");
		return -1;
	}

	inicializaMatrixFlag<<<grid,block>>>(matrixFlagDev, rows, columns);
	errCuda = cudaGetLastError();
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se inicializo matrixFlagDev . Saliendo...\n");
		return -1;
	}

	/* 3. Etiquetado inicial */
	etiquetadoInicialKernel<<<grid,block>>>(matrixDataDev, matrixResultDev, matrixResultCopyDev, rows, columns);
	errCuda = cudaGetLastError();
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("Fallo Kernel de etiquetado inicial. Saliendo...\n");
		return -1;
	}

	/* 4. Computacion */
	int t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	int flagCambio=1;
	// int *dflagCambio;
	// &dflagCambio = 1;

	/* 4.2 Busqueda de los bloques similiares */
	for(t=0; flagCambio !=0; t++){
		flagCambio=0;
		//&dflagCambio = 0;

		/* 4.2.1 Actualizacion copia */
		actualizacionCopiaKernel<<<grid,block>>>(matrixResultDev, matrixResultCopyDev, rows, columns);
		if(errCuda != cudaSuccess){
			printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
			printf("Fallo Kernel de actualizacionCopiaKernel, iteracion %d. Saliendo...\n", t);
			return -1;
		}

		/* 4.2.2 Computo y detecto si ha habido cambios */
		computationKernel<<<grid,block>>>(matrixDataDev, matrixResultDev, matrixResultCopyDev, matrixFlagDev, rows, columns);
		if(errCuda != cudaSuccess){
			printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
			printf("Fallo Kernel de computationKernel, iteracion %d. Saliendo...\n", t);
			return -1;
		}

		flagCambio = reduce(matrixFlagDev, rows*columns);


		// AHORRAR ESTO DE ABAJO
		//Transferencia de matrixFlag a host
	  // errCuda = cudaMemcpy(matrixFlag, matrixFlagDev, rows*columns* sizeof(int), cudaMemcpyDeviceToHost);
		// if(errCuda != cudaSuccess){
		// 	printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		// 	printf("No se copio matrixFlagDev a matrixFlag en iteracion %d. Saliendo...\n", t);
		// 	return -1;
		// }
		//
		// for(i=1;i<rows-1;i++){
		// 	for(j=1;j<columns-1;j++){
		// 		flagCambio += matrixFlag[i*columns+j];
		// 	}
		// }
		// AHORRAR ESTO DE ARRIBA


		printf("Iteracion %d, flagCambio=%d\n", t, flagCambio);

		#ifdef DEBUG
			printf("\nResultados iter %d: \n", t);
			for(i=0;i<rows;i++){
				for(j=0;j<columns;j++){
					printf ("%d\t", matrixResult[i*columns+j]);
				}
				printf("\n");
			}
		#endif

	}

	//Transferencia de matrixResult a host
	errCuda = cudaMemcpy(matrixResult, matrixResultDev, rows*columns* sizeof(int), cudaMemcpyDeviceToHost);
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("No se copio matrixResultDev a matrixResult. Saliendo...\n");
		return -1;
	}

	/* 4.3 Inicio cuenta del numero de bloques */
	numBlocks=0;
	for(i=1;i<rows-1;i++){
		for(j=1;j<columns-1;j++){
			if(matrixResult[i*columns+j] == i*columns+j) numBlocks++;
		}
	}

//
// EL CODIGO A PARALELIZAR TERMINA AQUI
//

	/* PUNTO DE FINAL DE MEDIDA DE TIEMPO */
	cudaDeviceSynchronize();
 	double t_fin = cp_Wtime();


	/* 5. Comprobación de resultados */
  	double t_total = (double)(t_fin - t_ini);

	printf("Result: %d:%d\n", numBlocks, t);
	printf("Time: %lf\n", t_total);
	#ifdef WRITE
		printf("Resultado: \n");
		for(i=0;i<rows;i++){
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixResult[i*columns+j]);
			}
			printf("\n");
		}
	#endif

	/* 6. Liberacion de memoria */
	free(matrixData);
	free(matrixResult);
	free(matrixResultCopy);
	free(matrixFlag);

	cudaFree(matrixDataDev);
	cudaFree(matrixResultDev);
	cudaFree(matrixResultCopyDev);
	cudaFree(matrixFlagDev);

	cudaDeviceReset();

	errCuda = cudaGetLastError();
	if(errCuda != cudaSuccess){
		printf("ErrCUDA: %s\n", cudaGetErrorString(errCuda));
		printf("Fallo al liberar recursos de GPU. Saliendo...\n");
		return -1;
	}

	return 0;
}
