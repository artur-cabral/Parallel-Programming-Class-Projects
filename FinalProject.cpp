//============================== FINAL PROJECT ====================================
// Author: Artur Cabral
// Final Project: C++11 Multithreading Programming - MonteCarlo Integration
//Summary: the goal for this final project is to learn how to program in multithread
// c++11 to parallelize the montecarlo integration
// Important routines: montecarlo();
//=================================================================================
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <mutex>
#include <ctime>

std::mutex mutex;
long num_threads = 128;
long N = 1000000;
long thread_counter, total_samples, total_sum;
//=========================== FUNCTIONS ===================================
// This function takes in a X value and returns the Y value after computing it
double montecarlo(double x){
double y;
y = (exp(x*x) - 1)/(exp(1) - 1);

return y;
}
//This function tells the thread to generate a random number, and compare to the
// compute the  y and compare to the random y
void threadwork(){
long sum = 0;
long samples = N/num_threads;
double y;
// Create the for loop from 0 to N to call the MonteCarlo Integration Function
for (int i=0;i<samples;i++){
// Get random X and Y coordinates between 0 and 1
//Plug X into the MonteCarlo Function
y = montecarlo((double) rand() / RAND_MAX);
//If the result is less than the random Y, count ++, if greater, don't count
if ((double) rand() / RAND_MAX <y)
	sum++;
	}//end for loop

{
std::lock_guard<std::mutex> guard( mutex );
total_sum += sum;
total_samples += samples;

}


}// end of function
//============================ MAIN ========================================
int main(){

total_samples = 0;
total_sum = 0;
std::clock_t begin = clock();
//Launch the group of threads
std::vector<std::thread> threads;
for(int i=0;i<num_threads;i++)
	threads.push_back(std::thread(threadwork));

//Wait for all threads to finish
for(auto& thread : threads)
	thread.join();
// Average all the points less than the line, and the percentage is the integral
double result;
result = (double) total_sum/total_samples;
std::clock_t end = clock();
double time_change = double(end-begin) / CLOCKS_PER_SEC;
std::cout<< "The result is " << result << std::endl;
std::cout<< "Time Change: " << time_change << " s " << std::endl;

}
