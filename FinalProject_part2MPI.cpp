//===================== FINAL PROJECT PART 2 ============================
// Author: Artur Cabral
// Final Project Part 2 - New Application(MPI)
// Summary: on this part 2 of the project, the goal is to create a routine
// for generic parallel numerical integration which uses the midpoint rule
// Important routines:
//=======================================================================
#include <iostream>
#include <mpi.h>
#include <stdio.h>
//assign values to a,b, and n
double a = 0;
double b = 5;
int n = 5;
double w,*f,*mid,sum,globalsum;
int source = 0;
//=================== FUNCTIONS ========================================
// b is the upper bound, a is the lower bound of the integral
// n is the number of slivers of equal width
// the width of each sliver is given by w = (b-a)/n
// then calculate the midpoint of each sliver
// each sliver area is given by w*f(x)
// the entire integral is given bu the sum of all these sliver areas estimates
// the midpoint of the kth sliver is given by the formula Mk = a + ((2k-1)/2n)*(b-a)
double midpoint(double a,double b,int localsize,int source){
sum=0;
globalsum=0;
//all processors compute the integral approximation
//calculate the width of each sliver
w = (b-a)/n;
//calculate the midpoints
for (int i=0;i<localsize;i++){
	mid[i] = a + ((double)(2*i - 1)/(double)(2*n))*(b-a);
//	std::cout<<mid[i]<<"  ";
}
//std::cout<<std::endl;
//calculate the f(mid)
for (int i=0;i<localsize;i++){
	f[i] = mid[i]*mid[i];
//	std::cout<<f[i]<< "  ";
}

// here, use whatever function you want, in this case I used x^2
//sum all f(mid)and multiply by the width of the sliver
for (int i=0;i<localsize;i++)
	sum += f[i]*w;
//use MPI_Reduce to get the double sum and send it to a global sum variable so it
//can be broadcasted to everyone
MPI_Reduce(&sum,&globalsum,1,MPI_DOUBLE,MPI_SUM,source,MPI_COMM_WORLD);

return globalsum;
}//end of midpoint routine

//================= MAIN ==============================================
int main(int argc, char *argv[]){
MPI_Init(&argc,&argv);
int rank, numprocs;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//allocate the array sizes
if(rank==source){
f = new double[n];
mid = new double[n];
}
else{
f = NULL;
mid = NULL;
}
MPI_Status status;

//allocate sizes and set which processor gets which part
int *localsize;
localsize = new int[numprocs];
for(int i=0;i<numprocs;i++)
	localsize[i] = n/numprocs;
for(int i=0;i<n%numprocs;i++)
	localsize[i]++;
//startpart array is a cumulative sum
int *startpart;
startpart = new int[numprocs];
startpart[0] = 0;
for (int i=0;i<numprocs;i++)
	startpart[i] = startpart[i-1] + localsize[i-1];
//allocate local arrays to their correct sizes
double *localF, *localMid;
localF = new double[localsize[rank]];
localMid = new double[localsize[rank]];
int tag;
//A special case, we don't pass from source to source, so we just set the processor
//source's local elements directly
if(rank==source){
for(int k=0;k<localsize[rank];k++){
	localF[k] = f[k];
	localMid[k] = mid[k];
	}
}
// Processor source sends parts of f and mid to all other processors using MPI_Scatterv
MPI_Scatterv(f,localsize,startpart,MPI_DOUBLE,localF,localsize[rank],MPI_DOUBLE,source,MPI_COMM_WORLD);
MPI_Scatterv(mid,localsize,startpart,MPI_DOUBLE,localMid,localsize[rank],MPI_DOUBLE,source,MPI_COMM_WORLD);

// call the routine and each processor computes their part and time it
double start = MPI_Wtime();

midpoint(a,b,localsize,source);

double end = MPI_Wtime();
//Broadcast globalsum to everyone

//Print result
if(rank==source)
	std::cout<< "Approximation: " << globalsum <<std::endl;

printf("Time Change: %.16g s \n",end-start);



}
