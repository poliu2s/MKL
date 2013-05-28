#include <iostream>
#include <windows.h>
#include <ctime>
#include <mkl.h>

using namespace std;

int main()
{	
	// Testing exponential calculation on matrix

	int temp_array_size = 9;

	double* A = (double*)mkl_malloc(3 * 3 * sizeof(double), 64);
	double* B = (double*)mkl_malloc(3 * 3 * sizeof(double), 64);

	A[0] = 1.0;
	A[1] = 1.0;
	A[2] = 0.0;
	A[3] = 0.0;
	A[4] = 0.0;
	A[5] = 2.0;
	A[6] = 0.0;
	A[7] = 0.0;
	A[8] =-1.0;

	// MKL function
	vdExp(temp_array_size, A, B);

	for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 3; j++)
			cout << B[i*3+j] << " ";
		cout << endl;
	}
	getchar();

	return 0;
}