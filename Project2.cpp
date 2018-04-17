//================================= PROJECT 2 =================================================
// Author: Artur Cabral
// Project 2: Matrix-Vector Multiplication in MPI
// Summary: in this project, the goal is to implement a distributed version of Matrix-Vector
// multiplication
// Important routines:
//=============================================================================================
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <vector>

//==================================== FUNCTIONS =======================================
void printMatrix(float *matrix, int N){
std::cout<< "= [ ";
for(int i=0;i<N;i++)
	std::cout<<matrix[i]<< " ";
std::cout<< " ]" << std::endl;
}//end of print matrix

void matrixVecMulti(float *myLocalB, float *myLocalA, float *xFull, int N, int rank, int *localsize){

for(int i=0;i<localsize[rank];i++){
	myLocalB[i] = 0;
	for(int j=0;j<N;j++){
	myLocalB[i] += myLocalA[i*N+j] * xFull[j];
	}
  }

}//end of matrixVecMulti

void matrixVecMulti_noA(float *myLocalB, float *xFull, int N, int rank, int numprocs, int *startpart, int *localsize){
int i=0;
if(rank==0)
	myLocalB[i] = 2*xFull[i] - xFull[i+1];
else
	myLocalB[i] = -xFull[startpart[rank]-1+i] + 2*xFull[startpart[rank]+i] - xFull[startpart[rank]+1+i];

i=localsize[rank];
if(rank==numprocs-1)
	myLocalB[localsize[rank]-1] = -xFull[N-2] + 2*xFull[N-1];
else
	myLocalB[i] = -xFull[startpart[rank]-1+i] + 2*xFull[startpart[rank]+i] - xFull[startpart[rank]+1+i];


for(int i=1;i<localsize[rank]-1;i++)
	myLocalB[i] = -xFull[startpart[rank]-1+i] + 2*xFull[startpart[rank]+i] - xFull[startpart[rank]+1+i];

}// end of matrixVecMulti_noA

//================================== MAIN ==============================================
int main(int argc, char *argv[]){
MPI_Init(&argc,&argv);
int rank, numprocs;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//create and allocate all the matrix A of size N^2 and the "vector" x
float *Afull, *xFull, *bFull;
int source = 0;
//define size??
int N = 16384;

if (rank==source){
	Afull = new float[N*N];
	xFull = new float[N];
	bFull = new float[N];
// fill A and x with values
//fill x with all ones
for(int i=0;i<N;i++)
	xFull[i] = 1;
//fill matrix A with specific values
int count = 0;
for(int i=0;i<N;i++){
	for(int j=0;j<N;j++){
		Afull[count] = 0;
		if(i==j)
			Afull[count] = 2;
		else if(i==j+1)
			Afull[count] = -1;
		else if (i==j-1)
			Afull[count] = -1;
		count++;
	}
    }
}
else {
	Afull = NULL;
	xFull = new float[N];
	bFull = new float[N];
//fill x with all ones
for(int i=0;i<N;i++)
	xFull[i] = 1;
}

MPI_Status status;

//allocate the sizes and set which processor gets which part
int *localsize;
localsize = new int[numprocs];
for(int i=0;i<numprocs;i++)
	localsize[i] = N/numprocs;
for(int i=0;i<N%numprocs;i++)
	localsize[i]++;
//allocate the size of the matrix
int *localsizeM;
localsizeM = new int[numprocs];
for (int i=0;i<numprocs;i++)
	localsizeM[i] = N*N / numprocs;
for(int i=0;i<N*N%numprocs;i++)
	localsizeM[i]++;
//startpart array is a cumulative sum
int *startpart;
startpart = new int[numprocs];
startpart[0] = 0;
for(int i=0;i<numprocs;i++)
	startpart[i] = startpart[i-1] + localsize[i-1];
//startpart for the matrix
int *startpartM;
startpartM = new int[N*numprocs];
startpartM[0] = 0;
for(int i=0;i<numprocs;i++)
	startpartM[i] = startpartM[i-1] + localsizeM[i-1];

//allocate local arraya to their correct sizes
float *myLocalA, *myLocalB;
myLocalA = new float[localsizeM[rank]];
myLocalB = new float[localsize[rank]];


// A special case, we don't pass from source to source, so we just set the
// processor source's local elements directly
if(rank==source){
for(int k=0;k<localsize[rank];k++){
	myLocalA[k] = Afull[k];

}
}
//call the routine and time it to evaluate and average performance
double start = MPI_Wtime();

//Processor source sends parts of A to all other processors using Scatter
MPI_Scatterv(Afull, localsizeM,startpartM,MPI_FLOAT,myLocalA,localsizeM[rank],MPI_FLOAT,source,MPI_COMM_WORLD);

//each processor does its own dot product using their part of A and the vector X
matrixVecMulti_noA(myLocalB,xFull,N,rank,numprocs,startpart,localsize);

//each processor sends its dot product calculation to the source processor using MPI_Gather
MPI_Gatherv(myLocalB, localsize[rank],MPI_FLOAT,bFull, localsize,startpart,MPI_FLOAT,source,MPI_COMM_WORLD);
//source processor broadcasts the resulting vector to all processors using MPI_Bcast
MPI_Bcast(bFull,N,MPI_FLOAT,source,MPI_COMM_WORLD);
double end = MPI_Wtime();

//Print the result

//printMatrix(bFull,N);
printf("Time Change: %.16g s \n",end-start);

MPI_Finalize();
}
