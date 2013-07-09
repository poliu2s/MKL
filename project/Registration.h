//Author: Changhoon Baek(45322096)
//Edited by: Po Liu (13623079)

#include <math.h>
#include "MultiObjModel.h"
#include "MultiObjModelMKL.h"
#include "USVolume.h"
#include <iostream>
#include "vnl/vnl_trace.h"
#include "vnl/vnl_transpose.h"
#include "vnl/vnl_matrix.h"
#include "vnl/vnl_diag_matrix.h"
#include "vnl/algo/vnl_svd.h"
#include "vnl/algo/vnl_determinant.h"
#include "vnl/algo/vnl_amoeba.h"
#include "vnl/vnl_vector.h"
#include "vnl/vnl_cost_function.h"
#include "vnl/vnl_inverse.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cstdio>


#ifndef REGISTRATION_H
#define REGISTRATION_H


class Registration {
public:
	Registration();
	Registration(MultiObjModel &model, double shape_coef_length, double pose_coef_length,
		         double gammaShapeCoef, double gamma_pose_coef);
	Registration(MultiObjModelMKL &model, double shape_coef_length, double pose_coef_length,
		         double gammaShapeCoef, double gamma_pose_coef);
	~Registration();
 
	void set_model_(MultiObjModel model);
	void set_rigid_tr_(vnl_matrix<double>* rigid_tr);
	void set_rigid_tr_MKL_(double* rigid_tr);
	void set_US_vol_(USVolume volume);
	void set_max_iter_(int maxIter);
	void set_sigma_(double sigma);
	void set_sigma_threshold_(double sigma_threshold);
	void set_gen_point_num_(int gen_point_num);
	void set_outlier_(double outlier);
	double get_gen_point_num_(void);

	void run(void);
	void runMKL(void);
	static void generateProbabilityMap(vnl_matrix<double> *x, vnl_matrix<double> *xPr,
		                               vnl_matrix<double> *y, double sigma2, double outlier,
									   vnl_matrix<double> *P1, vnl_matrix<double> *Pt1,
									   vnl_matrix<double> *Px);
	static void generateProbabilityMap2(vnl_matrix<double> *x, vnl_matrix<double> *xPr, vnl_matrix<double> *y,
		                                double sigma2, double outlier, vnl_matrix<double> *P1,
										vnl_matrix<double> *Pt1, vnl_matrix<double> *Px);
	static void generateProbabilityMap3(vnl_matrix<double> *x, vnl_matrix<double> *xPr, vnl_matrix<double> *y,
		                                double sigma2, double outlier, vnl_matrix<double> *P1,
										vnl_matrix<double> *Pt1, vnl_matrix<double> *Px);
	static void generateProbabilityMap4(double *x, int x_rows, int x_cols, double *xPr, double *y,
		                                int y_rows, double sigma2, double outlier,
										double *P1, double *Pt1, double *Px);
	static void generateProbabilityMap5(double *x, int x_rows, int x_cols, double *xPr, double *y,
		                                int y_rows, double sigma2, double outlier,
										double *P1, double *Pt1, double *Px);
	static void generateProbabilityMap6(double *x, int x_rows, int x_cols, double *xPr, double *y,
		                                int y_rows, double sigma2, double outlier,
										double *P1, double *Pt1, double *Px);

	static vnl_matrix<double> rigidRegistration(vnl_matrix<double> *x, vnl_matrix<double> *xPr,
		                                        vnl_matrix<double> *y, vnl_matrix<double> *y_init,
												double* sigma2, double outlier);
	static double* rigidRegistration2(double* x, int x_rows, int D,
												 double* xPr, 
										         double* y, int y_rows,
										         double* y_init, 
										         double* sigma, 
										         double outlier,
												 double* Px,
												 double* Pt1,
												 double* P1);
	static void optimizeShapeAndPoseCoef(vnl_vector<double>* x_min, double shapeCoefLengthd,
		                                 MultiObjModel *model,vnl_matrix<double> *Px,vnl_matrix<double> *P1,
										 vnl_matrix<double> *gammaPose,vnl_matrix<double> *gammaShape);
	static void optimizeShapeAndPoseCoefMKL(double* x_min, 
											int shapeCoefLength,
											int poseCoefLength,
											MultiObjModelMKL *model, 
											double* Px, 
											double* P1, 
											int M,
											double* gammaPose, 
											double* gammaShape);

	static void computeMultiTransformationValueAndDerivative(double* theta, double* basis, int L, double* poseMean, double* deriv, double* value);
	static void computePointCorrValueAndDerivative(double* poseCoef, int L, double* PX, double* P1,
												   double* poseMean, double* y, int N, double* basis, int pose_modes_nr_, double* deriv);
	static void multiObjectAtlasPoseCoefCostFunction(double* poseCoef, int L, MultiObjModelMKL* model,
													 double* shapeDeformed, double* PX,
													 double* P1, double Gamma, double* deriv);
	static void fill_mkl_matrix(double* matrix, int length, double fill_constant);
	


	//	vnl_matrix<double>* P1, vnl_matrix<double>* Pt1, vnl_matrix<double>* Px, vnl_matrix<double>* generatedPoint, vnl_matrix<double>* poseDeformed, vnl_matrix<double>* mu_generatedPoint, vnl_matrix<double>* mu_poseDeformed);

	//test//
	void set_generated_point_(vnl_matrix<double>* generated_point);
	void set_generated_point_prb_(vnl_matrix<double>* generated_point_prb);
	void set_mu_generated_point_(vnl_matrix<double>* mu_generated_point);
	void set_mu_pose_deformed_(vnl_matrix<double>* mu_pose_deformed);
	void set_P1_(vnl_matrix<double>* P1);
	void set_Pt1_(vnl_matrix<double>* Pt1);
	void set_Px_(vnl_matrix<double>* Px);
	double get_sigma_(void);
	double get_shape_coef_length_(void);
	double get_pose_coef_length_(void);
	vnl_matrix<double>* get_registered_model(void);
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
	vnl_matrix<double>* generated_point_; //test
	vnl_matrix<double>* generated_point_prb_; //test
	vnl_matrix<double>* shape_coef_;
	vnl_matrix<double>* pose_coef_;
	vnl_matrix<double>* mu_generated_point_; //test
	vnl_matrix<double>* mu_pose_deformed_; //test
	vnl_matrix<double>* Px_;
	vnl_matrix<double>* Pt1_;
	vnl_matrix<double>* P1_;
	vnl_matrix<double>* rigid_tr_;
	vnl_matrix<double> gamma_pose_;
	vnl_matrix<double> gamma_shape_;
	
	
	MultiObjModel *model_;
	USVolume US_vol_;

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