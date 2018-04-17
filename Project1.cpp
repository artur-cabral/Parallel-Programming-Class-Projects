//========================= PROJECT 1 =================================================
// Author: Artur Cabral
// Project 1: Basic Matrix Arithmetic
// Summary: the goal for this project is to write routines to perform matrix addition,
// matrix-vector multiplication, and matrx-matrix multiplication.
// Important routines:
//=====================================================================================

#include <iostream>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <algorithm>
int numThreads = 4;

//========================== FUNCTIONS =============================================
// zeros - this routine allocates space for an N x M matrix and set it to be all zeros
template <class T>
std::vector<std::vector<T> > zeros(int N, int M){
std::vector<std::vector<T> > zero;
std::vector<T> Z;
for(int i=0;i<N;i++){
	Z.clear();
	for(int j=0;j<M;j++){
		Z.push_back(0);
	}
	zero.push_back(Z); //push back the values of Z into the 2D vector zero
}
return zero;

}//end of zeros

// zeros_1D - this routine populates a 1D vector with all zeros
template <class T>
std::vector<T> zeros_1D(int N){
std::vector<T> zeros;
for (int i=0;i<N;i++)
	zeros.push_back(0);

return zeros;
} //end of zeros_1D

// ones - this routine allocates space for an N x M matrix and set it to be all ones
template <class T>
std::vector<std::vector<T> > ones(int N, int M){
std::vector<std::vector<T> > one;
std::vector<T> temp;
for(int i=0;i<N;i++){
	temp.clear();
	for(int j=0;j<M;j++){
		temp.push_back(1);
	}
	one.push_back(temp); // push back the values of temp into the 2D vector one
}
return one;

}//end of ones

// ones_1D - this routine populates a 1D vector with all ones
template <class T>
std::vector<T> ones_1D(int N){
std::vector<T> ones;
for (int i=0;i<N;i++)
	ones.push_back(0);

return ones;
} //end of ones_1D

// id - this routine allocates space for an N x M identity matrix
template <class T>
std::vector<std::vector<T> > id(int N, int M){
std::vector<std::vector<T> > idmatrix;
std::vector<T> temp;
for(int i=0;i<N;i++){
	temp.clear();
	for(int j=0;j<M;j++){
		if(i == j){
			temp.push_back(1);
		}
		else{
			temp.push_back(0);
		}
	}
	idmatrix.push_back(temp); // push back the values of temp into the 2D vector idmatrix

}
return idmatrix;

}//end of id

// random - this routine allocates space for a N x M matrix in which is randomly assigned
// zeros and ones
template <class T>
std::vector<std::vector<T> > random(int N, int M){
std::vector<std::vector<T> > randmatrix;
std::vector<T> temp;
for(int i=0;i<N;i++){
	temp.clear();
	for(int j=0;j<M;j++){
		temp.push_back((double)rand()/(RAND_MAX)); //pushes back a random number between 0 and 1 to the allocated space
		}
	randmatrix.push_back(temp);
	}
return randmatrix;

}//end of random

// random_1D - this routine populates a 1D vector with random numbers between zero and one
template <class T>
std::vector<T> random_1D(int N){
std::vector<T> random;
for (int i=0;i<N;i++)
	random.push_back((double)rand()/(RAND_MAX));

return random;
} //end of 1D_zeros


// addMatrices1 - in this routine, we add two matrices, but for each "outside" vector, the
// "inside" vector in ran in parallel
template <class T>
std::vector<std::vector<T> > addMatrices1(int a, std::vector<std::vector<T> > &A, int b, std::vector<std::vector<T> > &B){
std::vector<std::vector<T> > result=zeros<T>(A.size(),A[0].size());

for(int i=0;i<A.size();i++){

	#pragma omp parallel for schedule(static) num_threads(numThreads)
	for(int j=0;j<A[0].size();j++){
		result[i][j] = (a*A[i][j] + b*B[i][j]);
		}
	}
return result;
}// end of addMatrices1

// addMatrices2 - in this routine, we add two matrices, but for each "outside" vector ran in
// parallel, we run the "inside" vector not in parallel
template <class T>
std::vector<std::vector<T> > addMatrices2(int a, std::vector<std::vector<T> > &A, int b, std::vector<std::vector<T> > &B){
std::vector<std::vector<T> > result=zeros<T>(A.size(),A[0].size());

#pragma omp parallel for schedule(static) num_threads(numThreads)
for(int i=0;i<A.size();i++)
	for(int j=0;j<A[0].size();j++){
		result[i][j] = a*A[i][j] + b*B[i][j];
	}

return result;
} // end of addMatrices2

//  matrixVec1 - in this routine, each individual row-column multiplication is distributed
// (this routine uses the same structure as the previous dotProduct4 form lab3)
template <class T>
std::vector<T> matrixVec1(std::vector<std::vector<T> > A, std::vector<T> x){
std::vector<T> ax=zeros_1D<T>(A.size());

for (int i=0;i<A.size();i++){

	#pragma omp parallel for num_threads(numThreads)
	for(int j=0;j<A[0].size();j++){
		ax[i] += A[i][j]*x[j];
	}
}

return ax;
} //end of matrixVec1

// matrixVec2 - in this routine, the outside vector is ran in parallel
template <class T>
std::vector<T> matrixVec2(std::vector<std::vector<T> > A, std::vector<T> x){
std::vector<T> ax=zeros_1D<T>(A.size());

#pragma omp parallel for num_threads(numThreads)
for(int i=0;i<A.size();i++){
	ax[i]=0;
	for(int j=0;j<A[0].size();j++){
		ax[i] += A[i][j]*x[j];
	}

}

return ax;

}// end of matrixVec2

// matrixMulti - this routine multiplies two given matrices,A and B, as long as their dimensions match. If they
// don't, it returns an error to the user. A matrix MxP has to be multiplied by an PxN, therefore the result is
// a M x N matrix
template <class T>
std::vector<std::vector<T> > matrixMulti(std::vector<std::vector<T> > &A, std::vector<std::vector<T> > &B){
std::vector<std::vector<T> > result=zeros<T>(A.size(),A[0].size());

#pragma omp parallel for num_threads(numThreads)
for(int i=0;i<A.size();i++){
	for(int j=0;j<A[0].size();j++){
		result[i][j]= 0;
		for(int k=0;k<A[0].size();k++){
			result[i][j] += A[i][k]*B[k][j];
		}
	}
}

return result;
} // end of matrixMulti

// printMatrix - this routine is to print a determined matrix given by the user
template <class T>
void printMatrix(std::vector<std::vector<T> > A){
for(int i=0;i<A.size();i++){
	std::cout<< "[ ";
	for(int j=0;j<A[0].size();j++){
		std::cout<< A[i][j] << "  ";

	}
	std::cout<<"]"<< std::endl;
}
} // end of printMatrix

// printVector - this routine has the function to print any given vector by the user
template <class T>
void printVector(std::vector<T> A){
std::cout<< "[ ";
for(int i=0;i<A.size();i++)
	std::cout<< A[i] << "  ";
std::cout<< "]" << std::endl;

} // end of printVector


//================================ MAIN =============================================
int main (int argc, const char *argv[]){
std::vector<std::vector<double> > A;
std::vector<std::vector<double> > B;
std::vector<std::vector<double> > C;
std::vector<double> X;
int a=2, b=3;

// timing zone - this zone times the function being ran
double start = omp_get_wtime(); //gets the current time

A = ones<double>(2048,2048);
B = ones<double>(2048,2048);
//X = random_1D<double>(8);
C = matrixMulti(A,B);
double end = omp_get_wtime(); //gets the current time
//printMatrix(C);
printf("Time Change: %.16g s \n", end-start); //calculates and displays the time change between operations

} // end of MAIN
