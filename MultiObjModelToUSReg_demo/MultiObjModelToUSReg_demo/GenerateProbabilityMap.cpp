#include "Registration.h"
#include <mkl.h>
#include "stdafx.h"



//============================================================generateProbabilityMap with MKL=====//
void Registration::generateProbabilityMap4(double *x, int x_rows, int D,
										   double *xPr,
										   double *y, int y_rows,
		                                   double sigma2,
										   double outlier, 
										   double *P1,
										   double *Pt1,
										   double *Px)
{
	// Initialize N, M, and D from input
	int N = x_rows;
	int M = y_rows;
	int P1_rows = y_rows;
	int P1_cols = 1;
	double ksig, outlier_tmp, sp;

	// Lookup table for the exponential
	double* expTable = (double*)mkl_malloc(10000 * sizeof(double), 64);//new double[1000];
	for(int i=0;i < 10000;i++)
		expTable[i] = exp(-(double)i/1000);

	double* P = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	double* temp_x = (double *)mkl_malloc( D*1*sizeof( double ), 64 );
	


	// Set sizes of matrices P1,Pt and Pt1. Fill them with zeros.
	//P1 = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	for(int i = 0; i < M*1; i++) P1[i] = 0.0;
	
	//Pt1 = (double *)mkl_malloc( N*1*sizeof( double ), 64 );
	//fill_mkl_matrix(Pt1, M, 1, 0.0);
	
	//Px = (double *)mkl_malloc( M*D*sizeof( double ), 64 );
	//fill_mkl_matrix(Px, M*D, 0.0);
	for(int i = 0; i < M*D; i++) Px[i] = 0.0;

	ksig = -2.0 * sigma2;
	outlier_tmp = (outlier * M * pow(-ksig*3.14159265358979,0.5*D) )/((1-outlier)*N);   


	// Matrices used for main loop
	double* Mx1 = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	for(int i = 0; i < M; i++) Mx1[i] = 1.0;

	double* Q = (double *)mkl_malloc( M*3*sizeof( double ), 64 );
	for(int i = 0; i < M*3; i++) Q[i] = 0.0;

	double* F = (double *)mkl_malloc( M*1*sizeof( double ), 64 );
	for(int i = 0; i < M*1; i++) F[i] = 0.0;

	double* tempM = (double *)mkl_malloc( 3*1*sizeof( double ), 64 );
	for(int i = 0; i < 3; i++) tempM[i] = 1.0;

	double one = 1.0;
	double negone = -1.0;
	double zero = 0.0;
	double beta = 1.0;
	double alpha = 1.0;

	double* x_nth_row = (double *)mkl_malloc( D*sizeof( double ), 64 ); //x + n * D * sizeof(double);
	double* temp_array = (double *)mkl_malloc( 1*1*sizeof( double ), 64 );

	int temp_matrix_size;

	// Main loop going over two point sets to calculate the probability map.
	for(int n = 0; n < N; n++) {

		//Q = Mx1 * x->get_n_rows(n,1)
		for(int i = 0; i<D; i++)
			x_nth_row[i] = x[n*D+i];

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    M, D, 1, one, Mx1, 1, x_nth_row, D, zero, Q, D);
	

		// Q = Q - y
		temp_matrix_size = y_rows * D;
		vdsub(&temp_matrix_size, Q, y, Q); 
		
		//Q->apply(squarefunction)
		int Q_rows = y_rows;
		int Q_cols = D;
		temp_matrix_size = Q_rows*Q_cols;
		vdmul(&temp_matrix_size, Q, Q, Q);		

		//*F = ( *Q * *tempM / ksig)
		beta = 1.0 / ksig;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    Q_rows, 1, Q_cols, beta, Q, Q_cols, tempM, 1, zero, F, 1);

		//F->apply(expfunction)
		int F_rows = Q_rows;
		int F_cols = 1;
		temp_matrix_size = F_rows * F_cols;

		// Calcuate exponential through the lookup table
		for(int i=0;i < Q_rows;i++)
		{/*
			if(F[i] < -10)
				F[i] = 0;
			else
				F[i] = expTable[-(int)floor(F[i]*1000)];*/
			F[i] = exp(F[i]);
		}
		//vdExp(temp_matrix_size, F, F);	
		
		//sp = (Mx1->transpose()* *F).get(0,0);
		
		//temp_array[0] = 0.0;
		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 
                    1, 1, M, one, Mx1, 1, F, F_cols, zero, temp_array, 1);
		sp = temp_array[0];
		sp += outlier_tmp;

		//*P = (*F/sp) * xPrb->get(n, 0) = F*(sp/xPrb->get(n,0))
		double multiplier = 1/ sp * xPr[n];
		for(int i = 0; i < F_rows; i++) {
			P[i] = F[i] * multiplier;
		}
		
		//Pt1->put(n,0,(1 - outlier_tmp/sp) * xPrb->get(n,0))
		Pt1[n] = (1 - outlier_tmp/sp) * xPr[n];

		//*P1 = *P1 + (*P)
		temp_matrix_size = M * 1;
		vdadd(&temp_matrix_size, P1, P, P1);


		//*Px = *Px + *P*x->get_n_rows(n,1)
		alpha = 1;
		beta = 1;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    M, D, 1, one, P, 1, x_nth_row, D, one, Px, D);

	}

	mkl_free_buffers();
	//mkl_thread_free_buffers();
	mkl_free(expTable);
	mkl_free(P);
	mkl_free(temp_x);
	mkl_free(Mx1);
	mkl_free(Q);
	mkl_free(F);
	mkl_free(tempM);
	mkl_free(x_nth_row);
	mkl_free(temp_array);

	return;	
}

// Helper function for MKL implementation
void Registration::fill_mkl_matrix(double* matrix, int length, double fill_constant)
{
	for(int i = 0; i < length; i++) {
		matrix[i] = fill_constant;
	}

}
