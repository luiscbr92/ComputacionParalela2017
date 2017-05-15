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
#define MAX_THREADS 1024

__global__ void etiquetadoInicialKernel(int* matrixDataDev, int* matrixResultDev, int* matrixResultCopyDev, int i, int columns){

	/*Identificaciones necesarios*/
	int IDX_Thread		=	threadIdx.x;				//Identificacion del hilo en la dimension x
	int IDY_Thread		=	threadIdx.y;				//Identificacion del hilo en la dimension y

	int IDX_block		=	blockIdx.x;				//Identificacion del bloque en la dimension x
	int IDY_block		=	blockIdx.y;				//Identificacion del bloque en la dimension y

	int shapeBlock_X	=	blockDim.x;				//Numeros del bloques en la dimension x
	int shapeBlock_Y	=	blockDim.y;				//Numeros del bloques en la dimension y

	

	matrixResultCopyDev[i*(columns)+j]=-1;
	matrixResultDev[i*(columns)+j]=-1;
	// Si es 0 se trata del fondo y no lo computamos
	if(matrixDataDev[i*(columns)+j]!=0){
		matrixResultDev[i*(columns)+j]=i*(columns)+j;
	}
}



/**
* Funcion secuencial para la busqueda de mi bloque
*/
int computation(int x, int y, int columns, int* matrixData, int *matrixResult, int *matrixResultCopy){
	// Inicialmente cojo mi indice
	int result=matrixResultCopy[x*columns+y];
	if( result!= -1){
		//Si es de mi mismo grupo, entonces actualizo
		if(matrixData[(x-1)*columns+y] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[(x-1)*columns+y]);
		}
		if(matrixData[(x+1)*columns+y] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[(x+1)*columns+y]);
		}
		if(matrixData[x*columns+y-1] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[x*columns+y-1]);
		}
		if(matrixData[x*columns+y+1] == matrixData[x*columns+y])
		{
			result = min (result, matrixResultCopy[x*columns+y+1]);
		}

		// Si el indice no ha cambiado retorna 0
		if(matrixResult[x*columns+y] == result){ return 0; }
		// Si el indice cambia, actualizo matrix de resultados con el indice adecuado y retorno 1
		else { matrixResult[x*columns+y]=result; return 1;}

	}
	return 0;
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

	/* 3. Etiquetado inicial */

	error = cudaMalloc(&matrixDataDev, rows*columns* sizeof(int));
	if(error != cudaSuccess){
		printf("error en maloc matrixDataDev");
	}
  cudaMemcpy(matrixData, matrixDataDev, rows*columns* sizeof(int), cudaMemcpyHostToDevice);
  matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int));
  cudaMalloc(&matrixResultDev, rows*columns* sizeof(int));
  matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
  cudaMalloc(&matrixResultCopyDev, rows*columns* sizeof(int));
  if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
     perror ("Error reservando memoria");
       return -1;
  }
	dim3 block(BLOCK_DIM_COLUMNAS,BLOCK_DIM_FILAS);
	int num_col_grid, num_fil_grid;
	if( columns % BLOCK_DIM_COLUMNAS != 0)
		num_col_grid = columns/BLOCK_DIM_COLUMNAS+1;
	else
		num_col_grid = columns/BLOCK_DIM_COLUMNAS;
	if( rows % BLOCK_DIM_FILAS != 0)
		num_fil_grid = rows/BLOCK_DIM_FILAS+1;
	else
		num_fil_grid = rows/BLOCK_DIM_FILAS;

	//printf("F%d-C%d\n", num_fil_grid, num_col_grid);
dim3 grid(num_col_grid, num_fil_grid);

numBlocks = num_col_grid * num_fil_grid;
// printf ("%d\n",numBlocks);


		etiquetadoInicialKernel<<< grid, block>>>(matrixDataDev, matrixResultDev, matrixResultCopyDev, rows, columns);
	// return 0;

	/* 4. Computacion */
	int t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	int flagCambio=1;

	/* 4.2 Busqueda de los bloques similiares */
	for(t=0; flagCambio !=0; t++){
		flagCambio=0;

		/* 4.2.1 Actualizacion copia */
		for(i=1;i<rows-1;i++){
			for(j=1;j<columns-1;j++){
				if(matrixResult[i*(columns)+j]!=-1){
					matrixResultCopy[i*(columns)+j]=matrixResult[i*(columns)+j];
				}
			}
		}

		/* 4.2.2 Computo y detecto si ha habido cambios */
		for(i=1;i<rows-1;i++){
			for(j=1;j<columns-1;j++){
				flagCambio= flagCambio+ computation(i,j,columns, matrixData, matrixResult, matrixResultCopy);
			}
		}

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

	return 0;
}
