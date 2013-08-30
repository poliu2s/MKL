//Author: Changhoon Baek(45322096)
//Edited by: Po Liu (13623079)

#include <math.h>
#include "MultiObjModelMKL.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <ctime>


#ifndef REGISTRATION_H
#define REGISTRATION_H

using namespace std;

class Registration {
public:
	Registration();
	Registration(MultiObjModelMKL &model, double shape_coef_length, double pose_coef_length,
		         double gammaShapeCoef, double gamma_pose_coef);
	~Registration();
 

	void set_rigid_tr_MKL_(double* rigid_tr);
	void set_max_iter_(int maxIter);
	void set_sigma_(double sigma);
	void set_sigma_threshold_(double sigma_threshold);
	void set_gen_point_num_(int gen_point_num);
	void set_outlier_(double outlier);
	double get_gen_point_num_(void);


	static double squarefunction(double x);
	static double sqrtfunction(double x);
	static double expfunction(double x);


	static void generateProbabilityMap4(double *x, int x_rows, int x_cols, double *xPr, double *y,
		                                int y_rows, double sigma2, double outlier,
										double *P1, double *Pt1, double *Px);




	static void computeMultiTransformationValueAndDerivative(double* theta, double* basis, int L, double* poseMean, double* deriv, double* value);
	static void computePointCorrValueAndDerivative(double* poseCoef, int L, double* PX, double* P1,
												   double* poseMean, double* y, int N, double* basis, int pose_modes_nr_, double* deriv);
	static void multiObjectAtlasPoseCoefCostFunction(double* poseCoef, int L, MultiObjModelMKL* model,
													 double* shapeDeformed, double* PX,
													 double* P1, double Gamma, double* deriv);
	static void multiObjectAtlasShapeCoefFunction(double* poseCoef, double* shapeCoef, int L, int* objInd,
													 double* shapeFeature, double* poseFeature,
													 double* poseDeformed, double* movedShape,
													 int obj_nr, double* poseMean, int shapeFeatureCols,
													 double* PX, double* P1, double Gamma,
													 double* deriv);
	static void fill_mkl_matrix(double* matrix, int length, double fill_constant);
	


	//	vnl_matrix<double>* P1, vnl_matrix<double>* Pt1, vnl_matrix<double>* Px, vnl_matrix<double>* generatedPoint, vnl_matrix<double>* poseDeformed, vnl_matrix<double>* mu_generatedPoint, vnl_matrix<double>* mu_poseDeformed);

	//test//
	double get_sigma_(void);
	double get_shape_coef_length_(void);
	double get_pose_coef_length_(void);
	double* get_registered_model_MKL(void);
	//====//

	static void printMatrix(double* mat, int x, int y, string name);

private:
	int max_iter_;
	double outlier_;
	double sigma_;
	double sigma_threshold_;
	double gen_point_num_;
	int shape_coef_length_;
	int pose_coef_length_;
	

	// ---------------------------------------------------

	int shape_coef_MKL_rows_;
	int pose_coef_MKL_rows_;
	
	double* shape_coef_MKL_;
	double* pose_coef_MKL_;
	double* Px_MKL_;
	double* Pt1_MKL_;
	double* P1_MKL_;
	double* rigid_tr_MKL_;
	double* gamma_pose_MKL_;
	double* gamma_shape_MKL_;
	//double* generated_point_MKL_;
	//double* generated_point_prb_MKL_;

	MultiObjModelMKL *model_MKL_;

	
	
};

#endif