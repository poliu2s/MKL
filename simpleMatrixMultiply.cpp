// Author: Po Liu
// Date: May 16, 2013

#include <iostream>
#include <windows.h>
#include <ctime>
#include <mkl.h>


using namespace std;
int main()
{	
	// MKL matrix multiplier: multiplies a 3x2 matrx with a 2x3 matrix
	// and displays the result
	double *A, *B, *C;
	int m, n, k, i, j;
	double alpha, beta;

	

	m = 3, k = 2, n = 3;
	alpha = 1.0; beta = 0.0;

	A = (double *)mkl_malloc( m*k*sizeof( double ), 64 );
    B = (double *)mkl_malloc( k*n*sizeof( double ), 64 );
	C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );

	cout << "A: ";
	for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
		
		if (i%k == 0) cout << endl;
		cout << A[i];
    }
	cout << endl;

	cout << "B: ";
    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(i+2);
		
		if (i%n == 0) cout << endl;
		cout << B[i];
    }
	cout << endl;


	for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
	}
	

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, 3, B, k, beta, C, n);

	cout << "Finished MKL calculations." << endl;
	
	
	cout << "Result : " << endl;
	for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
		  cout << C[j+i*m] << " ";
      }
	  cout << endl;
    }

	cout << "=================" << endl;
	getchar();
	return;
		
}