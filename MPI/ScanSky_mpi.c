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
	int destination;
	MPI_Status stat;

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
	if(world_rank != 0){
		matrixData= (int *)malloc( rows*(columns) * sizeof(int) );
		if ( (matrixData == NULL)   ) {
			perror ("Error reservando memoria");
			return -1;
		}
	}
	for (int dest = 1; dest< world_size; dest++){
		//printf("Hola desde %d columnas %d filas %d\n", world_rank, columns, rows);
		if(world_rank == 0){
			printf("Envio desde %d a %d\n", 0, dest);
			MPI_Send(&matrixData, (rows)*(columns), MPI_INT, dest, dest, MPI_COMM_WORLD);
		}
		if(world_rank == dest){
			printf("Recibo desde %d a %d\n", dest, 0);
			MPI_Recv(&matrixData, (rows)*(columns), MPI_INT, 0, dest, MPI_COMM_WORLD, &stat);
			printf("Hecho en el proceso %d\n", dest);
		}
	}
	//MPI_Bcast(&matrixData, (rows)*(columns), MPI_INT, 0, MPI_COMM_WORLD);
	//if ( world_rank == 0 ) {

		/* 3. Etiquetado inicial */
		matrixResult= (int *)malloc( (rows)*(columns) * sizeof(int) );
		matrixResultCopy= (int *)malloc( (rows)*(columns) * sizeof(int) );
		if ( (matrixResult == NULL)  || (matrixResultCopy == NULL)  ) {
			perror ("Error reservando memoria");
			return -1;
		}
	//}
	for(i=0;i< rows; i++){
		if(world_rank == 0){
			destination = i % (world_size -1) +1;
			MPI_Recv(&matrixResult[i*(columns)], columns, MPI_INT, destination, i, MPI_COMM_WORLD, &stat);
		}
		if(world_rank == i % (world_size -1) +1){
			for(j=0;j< columns; j++){
				matrixResult[i*(columns)+j]=-1;
				// Si es 0 se trata del fondo y no lo computamos
				if(matrixData[i*(columns)+j]!=0){
					matrixResult[i*(columns)+j]=i*(columns)+j;
				}
			}
			MPI_Send(&matrixResult[i*(columns)], columns, MPI_INT, 0, i, MPI_COMM_WORLD);
		}
	}
	//MPI_Bcast(&matrixResult, (rows)*(columns), MPI_INT, 0, MPI_COMM_WORLD);


		/* 4. Computacion */
		int t=0;
		/* 4.1 Flag para ver si ha habido cambios y si se continua la ejecucion */
		int flagCambio=1;

		/* 4.2 Busqueda de los bloques similiares */
		for(t=0; flagCambio !=0; t++){
			flagCambio=0;

			/* 4.2.1 Actualizacion copia */
			// LA PARALELIZACIÓN DE ESTE BUCLE NO APORTA MEJORA. ESTE IF ES PARA ELEGIR SI PARALELIZAR O NO
			// DE HECHO, FALLA CON PARALELIZACION: SE VA A TIEMPO LIMITE. PUEDE SER QUE LA PARALELIZACION NO
			// ESTE BIEN PENSADA. REVISAR
			if(0){
				for(i=1;i<rows-1;i++){
					if(world_rank == 0){
						destination = i % (world_size -1) +1;
						MPI_Send(&matrixResult[i*(columns)], columns, MPI_INT, destination, i, MPI_COMM_WORLD);
						//MPI_Send(&matrixResultCopy[i*(columns)], columns, MPI_INT, destination, i, MPI_COMM_WORLD);
						MPI_Recv(&matrixResultCopy[i*(columns)], columns, MPI_INT, destination, i , MPI_COMM_WORLD, &stat);
					}
					if(world_rank == i % (world_size -1) +1){
						int result[columns], resultCopy[columns];
						//resultCopy[0] = -1; resultCopy[columns-1] = -1;
						MPI_Recv(&result, columns, MPI_INT, 0, i, MPI_COMM_WORLD, &stat);
						//MPI_Recv(&resultCopy, columns, MPI_INT, 0, i, MPI_COMM_WORLD, &stat);
						// TODO: OPTIMIZABLE SI QUITAMOS EL BUCLE FOR Y HACEMOS EL PASO DE MENSAJE result -> resultCopy
						for(j=1;j<columns-1;j++){
							resultCopy[j]=result[j];
						}
						MPI_Send(&resultCopy, columns, MPI_INT, 0, i, MPI_COMM_WORLD);
					}
				}
			}
			else{
				if(world_rank == 0){
					for(i=1;i<rows-1;i++)
						for(j=1;j<columns-1;j++)
							matrixResultCopy[i*(columns)+j]=matrixResult[i*(columns)+j];
				}
			}


			if(world_rank == 0){
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
			MPI_Bcast(&flagCambio, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}


		/* 4.3 Inicio cuenta del numero de bloques */
		numBlocks=0;
		int numBlocksProc = 0;
		for(i=1;i<rows-1;i++){
			if(world_rank == 0){
				destination = i % (world_size -1) +1;
				MPI_Send(&matrixResult[i*(columns)], columns, MPI_INT, destination, i, MPI_COMM_WORLD);
			}
			if(world_rank == i % (world_size -1) +1){
				int result[columns];
				MPI_Recv(&result, columns, MPI_INT, 0, i, MPI_COMM_WORLD, &stat);
				for(j=1;j<columns-1;j++){
					if(result[j] == i*columns+j) numBlocksProc++;
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
	}

	MPI_Finalize();
	return 0;
}
