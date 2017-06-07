/*
* Contar cuerpos celestes
*
* Asignatura Computación Paralela (Grado Ingeniería Informática)
* Código secuencial base
*
* @author Ana Moretón Fernández
* @author Eduardo Rodríguez Gutiez
* @version v1.3
*
* (c) 2017, Grupo Trasgo, Universidad de Valladolid
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include "cputils.h"
#include <mpi.h>


/* Substituir min por el operador */
#define min(x,y)    ((x) < (y)? (x) : (y))

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
	int *matrixData=NULL;
	int *matrixResult=NULL;
	int *matrixResultCopy=NULL;
	int numBlocks=-1;
	int world_rank = -1;
	int world_size = -1;
	double t_ini;
	int i,j;
	int worker;
	MPI_Status stat;
	MPI_Request req;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size (MPI_COMM_WORLD, &world_size);

	if ( world_rank == 0 ) {

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
		int valor;
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
			matrixData[(columns-1)*(columns)+i]=0;
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


		/* PUNTO DE INICIO MEDIDA DE TIEMPO */
		t_ini = cp_Wtime();
	}

	//
	// EL CODIGO A PARALELIZAR COMIENZA AQUI
	//

	MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&columns, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int pos_ini[world_size];
	int num_of_rows[world_size];
	int pini = 1;
	for(int proc_id = 0; proc_id < world_size; proc_id++){
		num_of_rows[proc_id] = (rows-2) / world_size;
		if(proc_id < (rows-2) % world_size){
			num_of_rows[proc_id]++;
		}
		pos_ini[proc_id] = pini;
		pini += num_of_rows[proc_id];
	}

	// Inicializo matrixData para todos los procesos que no sean el 0
	if(world_rank != 0){
		matrixData= (int *)malloc( (num_of_rows[world_rank]+2)*(columns) * sizeof(int) );
		if (matrixData == NULL) {
			perror ("Error reservando memoria");
			return -1;
		}
	}

	// Inicializo matrixResult y su copia para todos los procesos
	matrixResult= (int *)malloc( (num_of_rows[world_rank]+2)*(columns) * sizeof(int) );
	matrixResultCopy= (int *)malloc( (num_of_rows[world_rank]+2)*(columns) * sizeof(int) );
	if ((matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
		perror ("Error reservando memoria");
		return -1;
	}

	// Comparto el trocito de matrixData de cada proceso para que empiecen a trabajar cada uno por su cuenta
	if(world_rank == 0)
		for(int proc_id = 1; proc_id < world_size; proc_id++)
			MPI_Send(&matrixData[(pos_ini[proc_id]-1)*columns], (num_of_rows[proc_id]+2)*(columns), MPI_INT, proc_id, 0, MPI_COMM_WORLD);
	else
		MPI_Recv(&matrixData[0], (num_of_rows[world_rank]+2)*columns, MPI_INT, 0, 0, MPI_COMM_WORLD, &stat);

	/* 3. Etiquetado inicial */
	for(i = 0; i < num_of_rows[world_rank]+2; i++){
		for(j=0;j< columns; j++){
			matrixResult[i*(columns)+j]=-1;
			// Si es 0 se trata del fondo y no lo computamos
			if(matrixData[i*(columns)+j]!=0){
				matrixResult[i*(columns)+j]=(pos_ini[world_rank]+i-1)*(columns)+j;
			}
		}
	}

	#ifdef WRITE
		if(world_rank == 0){
			printf("Inicializacion mresult \n");
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixResult[j]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);

		for(int proc_id = 0; proc_id < world_size; proc_id++){
			if(world_rank == proc_id){
				for(i=1;i<num_of_rows[world_rank]+1;i++){
					for(j=0;j<columns;j++){
						printf ("%d\t", matrixResult[i*(columns)+j]);
					}
					printf("\n");
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	#endif
	/* 4. Computacion */
	int t=0;
	/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
	int flagCambio=1;
	int flagCambioProc;

	/* 4.2 Busqueda de los bloques similiares */
	for(t=0; flagCambio !=0; t++){
		flagCambio=0;

		/* 4.2.1 Actualizacion copia */
		for(i = 1; i < num_of_rows[world_rank]+2; i++){
			for(j=1;j< columns; j++){
				matrixResultCopy[i*(columns)+j]=matrixResult[i*(columns)+j];
			}
		}

		flagCambioProc = 0;
		/* 4.2.2 Computo y detecto si ha habido cambios */
		// Primero intercambio las filas con el proceso de arriba
		if(world_rank != 0){
			MPI_Send(&matrixResultCopy[columns], columns, MPI_INT, world_rank-1, 1, MPI_COMM_WORLD);
			MPI_Recv(&matrixResultCopy[0], columns, MPI_INT, world_rank-1, 2, MPI_COMM_WORLD, &stat);
		}
		// Despues intercambio las filas con el proceso de abajo
		if(world_rank != world_size -1){
			MPI_Recv(&matrixResultCopy[(num_of_rows[world_rank]+1)*columns], columns, MPI_INT, world_rank+1, 1, MPI_COMM_WORLD, &stat);
			MPI_Send(&matrixResultCopy[(num_of_rows[world_rank])*columns], columns, MPI_INT, world_rank+1, 2, MPI_COMM_WORLD);
		}
		for(i = 1; i < num_of_rows[world_rank]+2; i++){
			for(j=1;j< columns; j++){
				flagCambioProc = flagCambioProc + computation(i,j,columns, matrixData, matrixResult, matrixResultCopy);
			}
		}

		MPI_Allreduce(&flagCambioProc, &flagCambio, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

		#ifdef DEBUG
			MPI_Barrier(MPI_COMM_WORLD);
			if(world_rank == 0){
				printf("Iteracion %d mresult \n",t);
				for(j=0;j<columns;j++){
					printf ("%d\t", matrixResult[j]);
				}
				printf("\n");
			}
			MPI_Barrier(MPI_COMM_WORLD);

			for(int proc_id = 0; proc_id < world_size; proc_id++){
				if(world_rank == proc_id){
					for(i=1;i<num_of_rows[world_rank]+1;i++){
						for(j=0;j<columns;j++){
							printf ("%d\t", matrixResult[i*(columns)+j]);
						}
						printf("\n");
					}
				}
				MPI_Barrier(MPI_COMM_WORLD);
			}
			MPI_Barrier(MPI_COMM_WORLD);
		#endif
	}

	/* 4.3 Inicio cuenta del numero de bloques */
	numBlocks=0;
	int numBlocksProc = 0;
	for(i = 1; i < num_of_rows[world_rank]+1; i++){
		for(j=1;j< columns; j++){
			if(matrixResult[i*(columns)+j] == (pos_ini[world_rank]+i-1)*(columns)+j){
				numBlocksProc++;
			}
		}
	}
	MPI_Reduce(&numBlocksProc, &numBlocks, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	//
	// EL CODIGO A PARALELIZAR TERMINA AQUI
	//
	if ( world_rank == 0 ) {

		/* PUNTO DE FINAL DE MEDIDA DE TIEMPO */
		double t_fin = cp_Wtime();

		/* 5. Comprobación de resultados */
		double t_total = (double)(t_fin - t_ini);

		printf("Result: %d\n", numBlocks);
		printf("Time: %lf\n", t_total);
	}
	#ifdef WRITE
		MPI_Barrier(MPI_COMM_WORLD);
		if(world_rank == 0){
			printf("Resultado \n",t);
			for(j=0;j<columns;j++){
				printf ("%d\t", matrixResult[j]);
			}
			printf("\n");
		}
		MPI_Barrier(MPI_COMM_WORLD);

		for(int proc_id = 0; proc_id < world_size; proc_id++){
			if(world_rank == proc_id){
				for(i=1;i<num_of_rows[world_rank]+1;i++){
					for(j=0;j<columns;j++){
						printf ("%d\t", matrixResult[i*(columns)+j]);
					}
					printf("\n");
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
		}
		MPI_Barrier(MPI_COMM_WORLD);
	#endif

	if ( world_rank == 0 ) {
		/* 6. Liberacion de memoria */
		free(matrixData);
		free(matrixResult);
		free(matrixResultCopy);
	}

	MPI_Finalize();
	return 0;
}
