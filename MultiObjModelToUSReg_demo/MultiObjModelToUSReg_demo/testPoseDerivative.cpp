#include <iostream>
#include <fstream>
#include <windows.h>
#include <ctime>
#include "Registration.h"
#include <math.h>
#include <mkl.h>
#include "stdafx.h"

using namespace std;






int main()
{	
	int testResult = 0;
	int temp_array_size;
	double shapeCoefLength = 5;
	double poseCoefLength = 5;
	MultiObjModelMKL model;

	model.readModel();

	
	
	// [movedShape shapeDeformed] = transferMultiObjectAtlas(modelE, [], [0.1 0.2], eye(4));
	double* RigidTr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) RigidTr[i] = 0.0;
	for(int i = 0; i < 16; i+=5) RigidTr[i] = 1.0;

	double* shapeCoef = 0; //= (double*)mkl_malloc(2 * 1 * sizeof(double), 64);
	double* poseCoef = (double*)mkl_malloc(2 * 1 * sizeof(double), 64);
	poseCoef[0] = 0.1;
	poseCoef[1] = 0.2;


	model.transferModel(shapeCoef, 0, poseCoef, 2, RigidTr);

	cout <<"model transfer" << endl;
	// Moved shape is correct - transferModel working
	// Registration::printMatrix(model.moved_shape_, 7500, 3, "model.moved_shape_");


	// X = movedShape+2;
	double* twos = (double*)mkl_malloc(7500 * 3 * sizeof(double), 64);
	for(int i = 0; i < 7500*3; i++) twos[i] = 2.0;

	double* X = (double*)mkl_malloc(7500 * 3 * sizeof(double), 64);
	for(int i = 0; i < 7500; i++) X[i] = 0.0;
	temp_array_size = 7500 * 3;
	vdAdd(temp_array_size, model.moved_shape_, twos, X);

	// Xpr = ones(size(X,1),1);
	double* Xpr = (double*)mkl_malloc(7500 * 1 * sizeof(double), 64);
	for(int i = 0; i < 7500; i++) Xpr[i] = 1.0;
	

	int sigma = 2;
	double outlier = 0.01;


	// [P1,Pt1,PX]=generateProbabitlityMap(X, Xpr, movedShape, sigma^2, outlier);
	double* P1 = (double*)mkl_malloc(7500 * 1 * sizeof(double), 64);
	double* Pt1 = (double*)mkl_malloc(7500 * 1 * sizeof(double), 64);
	double* Px = (double*)mkl_malloc(7500 * 3 * sizeof(double), 64);
	
	for(int i = 0; i < 7500; i++) Pt1[i] = 0.0;

	Registration::generateProbabilityMap4(X, 7500, 3, Xpr, model.moved_shape_, 7500, sigma*sigma,
										  outlier, P1, Pt1, Px);
	
	//--------------Test for large data set generateProbabilityMap4----------------------------
	/*
	int errors = 0;
	int errors2 = 0;
	
	double* P1_truth = (double*)mkl_malloc(7500 * 1 * sizeof(double), 64);
	double* Pt1_truth = (double*)mkl_malloc(7500 * 1 * sizeof(double), 64);
	double* Px_truth = (double*)mkl_malloc(7500 * 3 * sizeof(double), 64);

	if (!readMatrixFromFile("C:/MultiObjModelToUSReg/Matlab/testPose_P1.txt", P1_truth, 7500, 1))
		cout << "Not reading." << endl;
	
	if (!readMatrixFromFile("C:/MultiObjModelToUSReg/Matlab/testPose_PX.txt", Px_truth, 7500, 3))
		cout << "Not reading." << endl;

	
	Registration::generateProbabilityMap4(X, 7500, 3, Xpr, model.moved_shape_, 7500, sigma*sigma,
										  outlier, P1, Pt1, Px);


	for(int i = 0; i < 7500; i++) {
		if(fabs(P1_truth[i] - P1[i]) > 0.01)
			errors++;
	}

	for(int i = 0; i < 7500 * 3; i++) {
		if(fabs(Px_truth[i] - Px[i]) > 0.01)
			errors2++;
	}

	cout << "P1: " << errors << endl;
	cout << "Px: " << errors2 << endl;
	getchar();
	

	//Registration::printMatrix(P1_truth, 7500, 1, "P1_truth");
	Registration::printMatrix(X, 7500, 3, "X");
	*/
	//-----------------------------------------------------------------------------------------


	// [value deriv] = multiObjectAtlasPoseCoefCostFunction([0.1; 0.2], modelE, shapeDeformed, PX, P1, 0);
	//double* value;
	double* deriv = (double*)mkl_malloc(2 * sizeof(double), 64);
	for(int i = 0; i < 2; i++) deriv[i] = 0.0;

	double* input1 = (double*)mkl_malloc(2 * 1 * sizeof(double), 64);
	input1[0] = 0.1;
	input1[1] = 0.2;
	
	cout << "Testing..." << endl;
	Registration::multiObjectAtlasPoseCoefCostFunction(input1, 2, &model, model.shape_deformed_, Px, P1, 0, deriv);	
	cout << "Finished Test." << endl;

	cout << "Final deriv: " << deriv[0] << " " << deriv[1] << endl;
	//getchar();
	
	if( fabs(deriv[0] - (-0.1066)) > 0.001 ) {
		testResult = 1;
	}
	else if( fabs(deriv[1] - (-0.0284)) > 0.001 ) {
		testResult = 1;
	}
		
	cout << "testResult: " << testResult << endl;
	getchar();

	return testResult;
}