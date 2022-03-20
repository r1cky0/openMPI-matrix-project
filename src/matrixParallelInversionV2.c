#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define DIMBUFF 1000000
#define TOL 0.0001


int *allocVector(int dim){
	int *array = (int *)malloc(dim * sizeof(int));
	
	return array;
}

float *allocVectorFloat(int dim){
	float *array = (float *)malloc(dim * sizeof(float));
	
	return array;
}

float **alloc(int rows, int cols)
{
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array= (float **)malloc(rows*sizeof(float*));
    int i;
    for (i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

void free1D(int *vector){
	free(vector);
}

void free1DFloat(float *vector){
	free(vector);
}

//Function which frees the memory allocated before for the given matrix
void free2D(float **mat){
	free(mat[0]);
   	free(mat);
}

//Function which reads a matrix from a file and return it
float **readMatrixFromFile(char *path, int *nrows, int *ncolumns) {
	float **matrix;
	FILE *f;
	char buffer[DIMBUFF];
	int rows, columns;
	
	//File opening
	if(!(f = fopen(path, "r"))) {
		puts("File not found");
		return NULL;
	}
	
	//Reading the first line and passing arguments to the main program
	fgets(buffer, sizeof(buffer), f);
	sscanf(buffer, "%d %d", &rows, &columns);
	*nrows = rows;
	*ncolumns = columns;
	
	//Matrix allocation, first rows the columns
	matrix = alloc(rows, columns);
	
	//Reading matrix values
	int i = 0, j = 0;
	int counter = 0;
	while(fgets(buffer, sizeof(buffer), f)) {
		sscanf(buffer, "%f", &matrix[i][j]);
		j++;
		counter++;
		
		//When j == columns a row is filled, so I reset j and read the following row
		if(j == columns) {
			j = 0;
			i++;
		}
	}
	
	fclose(f);
	
	//Check validity and return the result
	if(counter != rows * columns) {
		return NULL;
	}
	return matrix;
}

//Function which prints a matrix and put a \n at the end
void printMatrix(float **matrix, int nrows, int ncolumns) {
	for(int i = 0; i < nrows; i++) {
		for(int j = 0; j < ncolumns; j++) {
			printf("%.3f ", matrix[i][j]);
		}
		puts("");
	}
	puts("");
}


//Inversion of a matrix NxN using LU decomposition method
int main(int argc, char *argv[]) {
	//Checking the arguments
	if(argc != 2) {
		puts("You have to pass a file which contains the matrix");
		return -1;
	}
	
	MPI_Status status;
	int myRank, size, retVal;
	int q, r;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	int n, columns;
	float **A, **Ainv;
	int *P;
	float **matrixPortion;
	int tol = TOL;
	clock_t tic;
	clock_t toc;
	
	if (myRank == 0){
		//Reading matrix from file
		A = readMatrixFromFile(argv[1], &n, &columns);
		if(A == NULL) {
			printf("File %s is not correct\n", argv[1]);
			MPI_Finalize();
			return 0;
		}

		//Checking if A is a square matrix
		if(n != columns) {
			puts("This is not a square matrix");
			MPI_Finalize();
			return 0;
		}
		
		P = allocVector(n + 1);
		Ainv = alloc(n, n);
		
		for (int i = 0; i <= n; i++){
			P[i] = i;
		}
		
//		printMatrix(A, n, n);

		tic = clock();
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

	for (int col = 0; col < n - 1; col++) {
		// il processo 0 ha le matrici, ora calcola il massimo

		float maxA, absA;
		int imax, j;
		float *ptr;

		if (myRank == 0){
			maxA = 0;
			imax = col;
			for (int i = col; i < n; i++){
				if ((absA = fabs(A[i][col])) > maxA){
					maxA = absA;
					imax = i;
				}
			}
		} 
		
		// per ogni colonna abbiamo l'indice del pivot
		// ora da scambiare righe
		
		if (myRank == 0){
			if (maxA < tol){
				puts("Matrice degenerata");
				MPI_Finalize();
				return 0;
			}
			
			if (imax != col) {
		        //pivoting P
				j = P[col];
				P[col] = P[imax];
				P[imax] = j;

				//pivoting rows of A
		        ptr = A[col];
		        A[col] = A[imax];
		        A[imax] = ptr;
		        //counting pivots starting from N (for determinant)
		        P[n]++;
			}
		}
			
		
		// inizia LU DECOMPOSITION
		// spedizione di riga principale e una porzione di righe per processo
		int startIndex;
		float mainRow[n];
		
		
		if (myRank == 0){
			for (int i = 0; i < n; i++){
				mainRow[i] = A[col][i];
			}
		}
		MPI_Bcast(&mainRow, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		if (myRank == 0) {
			// + 1 per skippare la riga principale già broadcastata
			q = (n - col - 1) / size;
			r = (n - col - 1) % size;
			
			matrixPortion = alloc(q + r, n);
			
			int nuovoIndex = col + 1;
			for (int i = 0; i < q + r; i++){
				matrixPortion[i] = A[nuovoIndex];
				nuovoIndex++;
			}
			
			
			for (int p = 1; p < size; p++){
				for (int i = q * p + r + col + 1; i < q * p + r + col + 1 + q; i++){
					retVal = MPI_Send(&A[i][0], n, MPI_FLOAT, p, 555, MPI_COMM_WORLD);
				}
			}
		} else {
			q = (n - col - 1) / size;
			r = (n - col - 1) % size;	
			matrixPortion = alloc(q, n);	
		
			for (int i = 0; i < q; i++){		
				retVal = MPI_Recv(&matrixPortion[i][0], n, MPI_FLOAT, 0, 555, MPI_COMM_WORLD, &status);
			}
			
		}
		
		
		if (myRank == 0){
			for (int i = 0; i < q + r; i++){
				matrixPortion[i][col] /= mainRow[col];
				
				for (int k = col + 1; k < n; k++) {
					matrixPortion[i][k] -= matrixPortion[i][col] * mainRow[k];
				}
				
			}
		} else {
			for (int i = 0; i < q; i++){
				matrixPortion[i][col] /= mainRow[col];
				
				for (int k = col + 1; k < n; k++) {
					matrixPortion[i][k] -= matrixPortion[i][col] * mainRow[k];
				}
			}
		}
		
		startIndex = col + 1;
		
		if (myRank == 0){
			for (int i = 0; i < q + r; i++){
				for (int j = 0; j < n; j++){
					A[startIndex][j] = matrixPortion[i][j];
				}
				startIndex++;
			}
		}

		if (col < n - size){
			if (myRank != 0){
				for (int i = 0; i < q; i++){
					retVal = MPI_Send(&matrixPortion[i][0], n, MPI_FLOAT, 0, 555, MPI_COMM_WORLD);
				}
			}
			if (myRank == 0){
				for (int p = 1; p < size; p++) {
					for (int i = 0; i < q; i++){
						retVal = MPI_Recv(&A[col + 1 + r + p * q + i][0], n, MPI_FLOAT, p, 555, MPI_COMM_WORLD, &status);
					}
				}
			}
		}
	}

	//Fine LU Decomposition
	
	
	//fare check determinante	
		
	if (myRank == 0){
		
		toc = clock();
		printf("LU decomposition %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	
		float det = A[0][0];

 		for (int i = 1; i < n; i++){
        	det *= A[i][i];
        }

    	det = (P[n] - n) % 2 == 0 ? det : -det;
    	
    	if (det == 0){
			puts("La matrice non è invertibile");
			MPI_Finalize();
			return 0;
		}
	}
		

	// fine calcolo determinante

	//inversione
	// faccio in seriale le righe e in parallelo le colonne
	// ogni elemento dipende dagli elementi della stessa colonna di Ainv (deve essere seriale)
	// ogni elemento dipende dagli elementi della stessa riga di A (broadcasto e calcolo in parallelo)
	
	// analogo con l'altro ciclo for ma indici in ordine inverso
	
	
	if (myRank != 0){
		A = alloc(n, n);
		P = allocVector(n + 1);
	}
		
	if (myRank == 0){
	
		tic = clock();
	
		for (int p = 1; p < size; p++){
			for (int i = 0; i < n; i++){
				for (int j = 0; j < n; j++){
					retVal = MPI_Send(&A[i][j], 1, MPI_FLOAT, p, 555, MPI_COMM_WORLD);
				}
			}
		}
	}
	
	if (myRank != 0){
		for (int i = 0; i < n; i++){
			for (int j = 0; j < n; j++){
				retVal = MPI_Recv(&A[i][j], 1, MPI_FLOAT, 0, 555, MPI_COMM_WORLD, &status);
			}
		}

	}

	MPI_Bcast(&P[0], n + 1, MPI_INT, 0, MPI_COMM_WORLD);
	// tutti i processi hanno A' e P
	// ora dividere la matrice per colonne
	
	float **recvMat;
	
	q = n / size;
	r = n % size;
	
	if (myRank == 0){
		recvMat = alloc(n, q + r); 	
	} else {
		recvMat = alloc(n, q);
	}
	
	if (myRank == 0){
		for (int j = 0; j < q + r; j++){
			for (int i = 0; i < n; i++){
				recvMat[i][j] = P[i] == j ? 1.0 : 0.0;

				for (int k = 0; k < i; k++){
					recvMat[i][j] -= A[i][k] * recvMat[k][j];
				}
			}
			
			for (int i = n - 1; i >= 0; i--){
				for (int k = i + 1; k < n; k++){
					recvMat[i][j] -= A[i][k] * recvMat[k][j];
				}
				recvMat[i][j] /= A[i][i];
			}
		}
	}
	if (n >= size){
	
	int realColumn = q * myRank + r;
	
		if (myRank != 0){
			for (int j = 0; j < q; j++){
				for (int i = 0; i < n; i++){
					recvMat[i][j] = P[i] == realColumn ? 1.0 : 0.0;

					for (int k = 0; k < i; k++){
						recvMat[i][j] -= A[i][k] * recvMat[k][j];
					}
				}
				
				for (int i = n - 1; i >= 0; i--){
					for (int k = i + 1; k < n; k++){
						recvMat[i][j] -= A[i][k] * recvMat[k][j];
					}
					recvMat[i][j] /= A[i][i];
				}
				realColumn++;
			}
		}
	}
	// tutti i processi hanno la loro porzione di AInv
	// il processo 0 deve ricevere dagli altri processi e cambiare la sua variabile da recvMat a Ainv 
	// fine
	if (myRank == 0){
		for (int j = 0; j < q + r; j++){
			for (int i = 0; i < n; i++){
				Ainv[i][j] = recvMat[i][j];
			}
		}
	} 
	if (myRank != 0) {
		for (int j = 0; j < q; j++){
			for (int i = 0; i < n; i++){
				retVal = MPI_Send(&recvMat[i][j], 1, MPI_FLOAT, 0, 555, MPI_COMM_WORLD);
			}
		}
	}
	
	if (myRank == 0){
		for (int p = 1; p < size; p++){
			for (int j = q * p + r; j < q * p + r + q; j++){
				for (int i = 0; i < n; i++){
					retVal = MPI_Recv(&Ainv[i][j], 1, MPI_FLOAT, p, 555, MPI_COMM_WORLD, &status);
				}
			}
		}
	}
	


	if (myRank == 0){
		
		toc = clock();
		printf("A' inversion %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
		
//		printMatrix(Ainv, n, n);
		free1D(P);
		free2D(Ainv);
	}
	
	if (myRank != 0){
		free2D(A);
	}
	MPI_Finalize();
	
	return 0;	
}




