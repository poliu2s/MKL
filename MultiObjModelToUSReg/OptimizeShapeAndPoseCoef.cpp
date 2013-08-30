#include "Registration.h"
#include <mkl.h>




//==================================================================OptimizeShapeAndPoseCoef=====//
void Registration::optimizeShapeAndPoseCoef(vnl_vector<double>* x_min,
											double shapeCoefLength,
											MultiObjModel *model,
											vnl_matrix<double> *Px,
											vnl_matrix<double> *P1,
											vnl_matrix<double> *gammaPose,
											vnl_matrix<double> *gammaShape)
{
	struct functor : public vnl_cost_function
	{
		vnl_vector<double> *x_min;
		vnl_matrix<double> *P1;
		vnl_matrix<double> *Px;
		
		double shapeCoefLength;
		
		MultiObjModel *model;
		vnl_matrix<double> *GammaShape;
		vnl_matrix<double> *GammaPose;

		double f(vnl_vector<double> const& params)
		{
			double value;
			
			vnl_matrix<double>* poseCoef = new vnl_matrix<double>; //(x->size() - shapeCoefLength, 1);
			vnl_matrix<double>* shapeCoef = new vnl_matrix<double>; //(shapeCoefLength, 1);
			vnl_matrix<double>* identity = new vnl_matrix<double>;
			vnl_matrix<double> movedShapeSum;
			vnl_matrix<double> sumMask(3, 1, 1);
			vnl_matrix<double> tmpSq;
			vnl_matrix<double> tmpMul;

			poseCoef->set_size(params.size() - shapeCoefLength, 1);
			shapeCoef->set_size(shapeCoefLength, 1);
			shapeCoef->set_column(0,params.extract(shapeCoefLength, 0));
			poseCoef->set_column(0, params.extract(params.size() - shapeCoefLength, shapeCoefLength));

			identity->set_size(4, 4);
			identity->set_identity();
			model->transferModel(shapeCoef, poseCoef, identity);
			model->extractVisiblePoints();

			tmpSq.set_size(model->moved_shape_visible_->rows(), model->moved_shape_visible_->cols());
			tmpMul.set_size(1, 1);
			tmpSq = model->moved_shape_visible_->apply(squarefunction);
			movedShapeSum = tmpSq * sumMask;	

			tmpMul = vnl_trace(Px->transpose() * *model->moved_shape_visible_); // Px = 5x3, model.movedShape = 7500x3 // (3x5) * (7500x3) = error

			value = (P1->transpose() * movedShapeSum).get(0,0) - 2*tmpMul.get(0,0) + (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0) + (poseCoef->transpose()**GammaPose**poseCoef).get(0,0); 

			free ((void*) poseCoef); //(x->size() - shapeCoefLength, 1);
			free ((void*) shapeCoef); //(shapeCoefLength, 1);
			free ((void*) identity);
	
			return value;
		}

		functor() : vnl_cost_function(1) {}
	};


	functor f;					
	f.x_min = new vnl_vector<double> ();
	f.Px = new vnl_matrix<double> (*Px);
	f.P1 = new vnl_matrix<double> (*P1);
	f.GammaPose = new vnl_matrix<double> (*gammaShape);
	f.GammaShape = new vnl_matrix<double> (*gammaPose);
	f.x_min = x_min;
	f.shapeCoefLength = shapeCoefLength;
	f.model = model;
	
	//f.Px = Px;
	//f.P1 = P1;
	
	f.GammaShape = gammaShape;
	f.GammaPose = gammaPose;
	vnl_amoeba minimizer(f);	

	minimizer.set_max_iterations(1000);					
	minimizer.set_x_tolerance(0.01);
	//minimizer.set_f_tolerance(0.001);
	minimizer.minimize(*x_min);	
}

//==================================================================OptimizeShapeAndPoseCoefMKL=====//
void Registration::optimizeShapeAndPoseCoefMKL(double* x_min, 
											   int shapeCoefLength,
											   int poseCoefLength,
											   MultiObjModelMKL *model, 
											   double* Px, 
											   double* P1, 
											   int M,
											   double* gammaPose, 
											   double* gammaShape)
{
	struct functor : public vnl_cost_function
	{
		double *x_min;
		double *P1;
		double *Px;
		
		int shapeCoefLength;
		int poseCoefLength;
		int M;
		
		MultiObjModelMKL *model;
		double *GammaShape;
		double *GammaPose;

		double f(vnl_vector<double> const& params)
		{
			//cout << "params: " << params[0] << endl;
			//getchar();
			double value;
			double one = 1.0;
			double three = 3.0;
			double zero = 0.0;
			
			double* poseCoef = (double*)mkl_malloc((params.size() - shapeCoefLength) * 1 * sizeof(double), 64);
			double* shapeCoef = (double*)mkl_malloc(shapeCoefLength * 1 * sizeof(double), 64);
			
			// shapeCoef->set_column(0,params.extract(shapeCoefLength, 0));
			for(int i = 0; i < shapeCoefLength; i++)
				shapeCoef[i] = params(i);

			// poseCoef->set_column(0, params.extract(params.size() - shapeCoefLength, shapeCoefLength));
			int len = params.size() - shapeCoefLength;
			for(int i = 0; i < len; i++)
				poseCoef[i] = params(shapeCoefLength + i);

			double* identity = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
			for(int i = 0; i < 16; i++) identity[i] = 0.0;
			for(int i = 0; i < 16; i+=5) identity[i] = 1.0;

			// model->transferModel(shapeCoef, poseCoef, identity);
			model->transferModel(shapeCoef, shapeCoefLength, poseCoef, poseCoefLength, identity);
			model->extractVisiblePoints();
			
			// tmpSq.set_size(model->moved_shape_visible_->rows(), model->moved_shape_visible_->cols());
			double* tmpSq = (double*)mkl_malloc(model->get_visible_point_nr_() * 3 * sizeof(double), 64);

			
			// tmpSq = model->moved_shape_visible_->apply(squarefunction);
			vdMul(model->get_visible_point_nr_() * 3, model->moved_shape_visible_, model->moved_shape_visible_, tmpSq);
			
			// vnl_matrix<double> sumMask(3, 1, 1);
			double* sumMask = (double*)mkl_malloc(3 * 1 * sizeof(double), 64);
			for (int i = 0; i < 3; i++)
				sumMask[i] = 1.0;

			// movedShapeSum = tmpSq * sumMask;
			double* movedShapeSum = (double*)mkl_malloc(model->get_visible_point_nr_() * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						model->get_visible_point_nr_(), 1, 3, one, tmpSq, 3, sumMask, 1, zero, movedShapeSum, 1);

			// tmpMul.set_size(1, 1);
			// tmpMul = vnl_trace(Px->transpose() * *model->moved_shape_visible_); // Px = 5x3, model.movedShape = 7500x3 // (3x5) * (7500x3) = error
			double* tmpMul = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			double* tmpMul1 = (double*)mkl_malloc(3 * 3 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,			
						3, 3, M, one, Px, 3, model->moved_shape_visible_, 3, zero, tmpMul1, 3);
			tmpMul[0] = 0.0;
			for(int i = 0; i < 9; i+=4)
				tmpMul[0] += tmpMul1[i];


			// value = (P1->transpose() * movedShapeSum).get(0,0) - 2*tmpMul.get(0,0) + (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0) + (poseCoef->transpose()**GammaPose**poseCoef).get(0,0); 
			//		1. (P1->transpose() * movedShapeSum).get(0,0)
			double* value_tmp1 = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, 1, M, one, P1, 1, movedShapeSum, 1, zero, value_tmp1, 1);
			value = value_tmp1[0];

			//		2. 2*tmpMul.get(0,0)
			value -= 2 * tmpMul[0];

			//		3. (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0)
			double* value_tmp2 = (double*)mkl_malloc(1 * (params.size() - shapeCoefLength) * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, (params.size() - shapeCoefLength), shapeCoefLength, one, shapeCoef, 1, GammaShape, (params.size() - shapeCoefLength), zero, value_tmp2, (params.size() - shapeCoefLength));
				
			double* value_tmp2a = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, 1, shapeCoefLength, one, value_tmp2, (params.size() - shapeCoefLength), shapeCoef, 1, zero, value_tmp2a, 1);
			value += value_tmp2a[0];
			
			//		4. (poseCoef->transpose()**GammaPose**poseCoef).get(0,0)
			double* value_tmp3 = (double*)mkl_malloc(1 * poseCoefLength * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, poseCoefLength, (params.size() - shapeCoefLength), one, poseCoef, (params.size() - shapeCoefLength), GammaPose, poseCoefLength, zero, value_tmp3, poseCoefLength);
			
			double* value_tmp3a = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, 1, (params.size() - shapeCoefLength), one, value_tmp3, poseCoefLength, poseCoef, 1, zero, value_tmp3a, 1);

	
			value += value_tmp3a[0];
			
			mkl_free_buffers();
			mkl_free(poseCoef);
			mkl_free(shapeCoef);
			mkl_free(identity);
			mkl_free(tmpSq);
			mkl_free(tmpMul);
			mkl_free(tmpMul1);
			mkl_free(sumMask);
			mkl_free(movedShapeSum);
			mkl_free(value_tmp1);
			mkl_free(value_tmp2);
			mkl_free(value_tmp2a);
			mkl_free(value_tmp3);
			mkl_free(value_tmp3a);
	
			return value;
		}

		functor() : vnl_cost_function(1) {}
	};

	functor f;	
	f.Px = Px;
	f.P1 = P1;
	f.x_min = x_min;

	
	f.GammaPose = gammaPose;
	f.GammaShape = gammaShape;

	f.shapeCoefLength = shapeCoefLength;
	f.poseCoefLength = poseCoefLength;
	f.M = M;
	f.model = model;

	// Minimization steps with VNL_Amoeba minimizer
	vnl_amoeba minimizer(f);	

	minimizer.set_max_iterations(1000);					
	minimizer.set_x_tolerance(0.01);
	minimizer.set_f_tolerance(0.001);

	vnl_vector<double>* x_min_local = new vnl_vector<double>(shapeCoefLength + poseCoefLength);
	for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
		x_min_local->put(i, x_min[i]);
	}
	minimizer.minimize(*x_min_local);

	for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
		x_min[i] = x_min_local->get(i);
	}

	free ((void*) x_min_local);
}

//==================================================================OptimizeShapeAndPoseCoefMKL_lbfgs=====//
void Registration::optimizeShapeAndPoseCoefMKL_lbfgs(double* x_min, 
													 int shapeCoefLength,
													 int poseCoefLength,
													 MultiObjModelMKL *model, 
													 double* Px, 
													 double* P1, 
													 int M,
													 double* gammaPose, 
													 double* gammaShape)
{

	
	struct functor : public vnl_cost_function
	{
		double *x_min;
		double *P1;
		double *Px;
		
		// For gradf function, we need separate expanded versions
		double *P1_large;
		double *Px_large;

		int shapeCoefLength;
		int poseCoefLength;
		int M;
		
		MultiObjModelMKL *model;
		double *GammaShape;
		double *GammaPose;

		double f(vnl_vector<double> const& params)
		{
			cout << "params: " << params[0] << endl;
			getchar();

			double value;
			double one = 1.0;
			double three = 3.0;
			double zero = 0.0;
			
			double* poseCoef = (double*)mkl_malloc((params.size() - shapeCoefLength) * 1 * sizeof(double), 64);
			double* shapeCoef = (double*)mkl_malloc(shapeCoefLength * 1 * sizeof(double), 64);
			
			// shapeCoef->set_column(0,params.extract(shapeCoefLength, 0));
			for(int i = 0; i < shapeCoefLength; i++)
				shapeCoef[i] = params(i);

			// poseCoef->set_column(0, params.extract(params.size() - shapeCoefLength, shapeCoefLength));
			int len = params.size() - shapeCoefLength;
			for(int i = 0; i < len; i++)
				poseCoef[i] = params(shapeCoefLength + i);

			double* identity = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
			for(int i = 0; i < 16; i++) identity[i] = 0.0;
			for(int i = 0; i < 16; i+=5) identity[i] = 1.0;

			// model->transferModel(shapeCoef, poseCoef, identity);
			model->transferModel(shapeCoef, shapeCoefLength, poseCoef, poseCoefLength, identity);
			model->extractVisiblePoints();
			
			// tmpSq.set_size(model->moved_shape_visible_->rows(), model->moved_shape_visible_->cols());
			double* tmpSq = (double*)mkl_malloc(model->get_visible_point_nr_() * 3 * sizeof(double), 64);

			
			// tmpSq = model->moved_shape_visible_->apply(squarefunction);
			vdMul(model->get_visible_point_nr_() * 3, model->moved_shape_visible_, model->moved_shape_visible_, tmpSq);
			
			// vnl_matrix<double> sumMask(3, 1, 1);
			double* sumMask = (double*)mkl_malloc(3 * 1 * sizeof(double), 64);
			for (int i = 0; i < 3; i++)
				sumMask[i] = 1.0;

			// movedShapeSum = tmpSq * sumMask;
			double* movedShapeSum = (double*)mkl_malloc(model->get_visible_point_nr_() * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						model->get_visible_point_nr_(), 1, 3, one, tmpSq, 3, sumMask, 1, zero, movedShapeSum, 1);

			// tmpMul.set_size(1, 1);
			// tmpMul = vnl_trace(Px->transpose() * *model->moved_shape_visible_); // Px = 5x3, model.movedShape = 7500x3 // (3x5) * (7500x3) = error
			double* tmpMul = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			double* tmpMul1 = (double*)mkl_malloc(3 * 3 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,			
						3, 3, M, one, Px, 3, model->moved_shape_visible_, 3, zero, tmpMul1, 3);
			tmpMul[0] = 0.0;
			for(int i = 0; i < 9; i+=4)
				tmpMul[0] += tmpMul1[i];


			// value = (P1->transpose() * movedShapeSum).get(0,0) - 2*tmpMul.get(0,0) + (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0) + (poseCoef->transpose()**GammaPose**poseCoef).get(0,0); 
			//		1. (P1->transpose() * movedShapeSum).get(0,0)
			double* value_tmp1 = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, 1, M, one, P1, 1, movedShapeSum, 1, zero, value_tmp1, 1);
			value = value_tmp1[0];

			//		2. 2*tmpMul.get(0,0)
			value -= 2 * tmpMul[0];

			//		3. (shapeCoef->transpose()**GammaShape**shapeCoef).get(0,0)
			double* value_tmp2 = (double*)mkl_malloc(1 * (params.size() - shapeCoefLength) * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, (params.size() - shapeCoefLength), shapeCoefLength, one, shapeCoef, 1, GammaShape, (params.size() - shapeCoefLength), zero, value_tmp2, (params.size() - shapeCoefLength));
				
			double* value_tmp2a = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, 1, shapeCoefLength, one, value_tmp2, (params.size() - shapeCoefLength), shapeCoef, 1, zero, value_tmp2a, 1);
			value += value_tmp2a[0];
			
			//		4. (poseCoef->transpose()**GammaPose**poseCoef).get(0,0)
			double* value_tmp3 = (double*)mkl_malloc(1 * poseCoefLength * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
						1, poseCoefLength, (params.size() - shapeCoefLength), one, poseCoef, (params.size() - shapeCoefLength), GammaPose, poseCoefLength, zero, value_tmp3, poseCoefLength);
			
			double* value_tmp3a = (double*)mkl_malloc(1 * 1 * sizeof(double), 64);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						1, 1, (params.size() - shapeCoefLength), one, value_tmp3, poseCoefLength, poseCoef, 1, zero, value_tmp3a, 1);

	
			value += value_tmp3a[0];
			
			mkl_free_buffers();
			mkl_free(poseCoef);
			mkl_free(shapeCoef);
			mkl_free(identity);
			mkl_free(tmpSq);
			mkl_free(tmpMul);
			mkl_free(tmpMul1);
			mkl_free(sumMask);
			mkl_free(movedShapeSum);
			mkl_free(value_tmp1);
			mkl_free(value_tmp2);
			mkl_free(value_tmp2a);
			mkl_free(value_tmp3);
			mkl_free(value_tmp3a);

			//cout << "value: " << value << endl;
			//getchar();
	
			return value;
		}

		void gradf(vnl_vector<double> const&x, vnl_vector<double> &gradient)
		{
			cout << "x(grad): " << x << endl;
			getchar();

			cout << "gradient: " << gradient << endl;
			getchar();

			double value;
			double one = 1.0;
			double three = 3.0;
			double zero = 0.0;
			
			double* poseCoef = (double*)mkl_malloc((x.size() - shapeCoefLength) * 1 * sizeof(double), 64);
			double* shapeCoef = (double*)mkl_malloc(shapeCoefLength * 1 * sizeof(double), 64);
			
			// shapeCoef->set_column(0,params.extract(shapeCoefLength, 0));
			for(int i = 0; i < shapeCoefLength; i++)
				shapeCoef[i] = x(i);

			// poseCoef->set_column(0, params.extract(params.size() - shapeCoefLength, shapeCoefLength));
			int len = x.size() - shapeCoefLength;
			for(int i = 0; i < len; i++)
				poseCoef[i] = x(shapeCoefLength + i);
			
			//printMatrix(P1_large, 7500, 1, "P1_before");
			//printMatrix(Px_large, 7500, 3, "PX_before");

			double* gamma_local = (double*)mkl_malloc(2 * 2 * sizeof(double), 64);
			for(int i = 0; i < 4; i++) {
				gamma_local[i] = 0.0;
			}

			//cout << "fails" << endl;
			//getchar();

			//printMatrix(poseCoef, 1, poseCoefLength, "poseCoef");
			//printMatrix(shapeCoef, 1, shapeCoefLength, "shapeCoef");
			

			// Calling shapeDeriv
			double* shapeDeriv = (double*)mkl_malloc(2 * sizeof(double), 64);
			multiObjectAtlasShapeCoefFunction(poseCoef, shapeCoef, shapeCoefLength, 
											  model->get_obj_ind_(), 
											  model->get_shape_feature_(), 
											  model->get_pose_feature_(), 
											  model->pose_deformed_, 
											  model->moved_shape_, 
											  model->get_obj_nr_(), 
											  model->get_pose_mean_(),
											  model->get_shape_modes_nr_(),
											  Px_large, 
											  P1_large, 
											  gamma_local, 
											  shapeDeriv);	

			printMatrix(shapeDeriv, 1, 2, "shapeDeriv");
			//printMatrix(model->shape_deformed_, 7500, 3, "model->shape_deformed_");

			

			// Calling poseDeriv
			double* poseDeriv = (double*)mkl_malloc(2 * sizeof(double), 64);
			multiObjectAtlasPoseCoefCostFunction(poseCoef, 
												 poseCoefLength, 
												 model, 
												 model->shape_deformed_, 
												 Px_large, 
												 P1_large, 
												 gamma_local, 
												 poseDeriv);

			printMatrix(poseDeriv, 1, 2, "poseDeriv");


			// Stitch the gradient cost back together
			gradient.set_size(shapeCoefLength + poseCoefLength);
			for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
				if(i < shapeCoefLength) {
					gradient[i] = shapeDeriv[i];
				} else {
					gradient[i] = poseDeriv[i - shapeCoefLength];
				}
			}

			cout << "gradient: " << gradient << endl;
			cout << "x: " << x << endl;
			getchar();
		}


		//functor() : vnl_cost_function(int num_unknowns) { set_number_of_unknowns(num_unknowns); }
		functor(int num_unknowns)  { set_number_of_unknowns(num_unknowns); }
	};

	const int num_unknowns = shapeCoefLength + poseCoefLength;
	cout << "num_unknowns: " << num_unknowns << endl;
	getchar();
	functor f(num_unknowns);
	f.Px = Px;
	f.P1 = P1;
	f.x_min = x_min;
	//f.set_number_of_unknowns(;
	
	f.GammaPose = gammaPose;
	f.GammaShape = gammaShape;

	f.shapeCoefLength = shapeCoefLength;
	f.poseCoefLength = poseCoefLength;
	f.M = M;
	f.model = model;

	// New matrix expansion for Px_large and P1_large
	double* Px_large = (double*)mkl_malloc(7500 * 3 * sizeof(double), 64);
	double* P1_large = (double*)mkl_malloc(7500 * 1 * sizeof(double), 64);
	for(int i = 0; i < 7500; i++) {
		for(int j = 0; j < 3; j++)
			Px_large[i*3 + j] = 0.0;
		P1_large[i] = 0.0;
	}
	

	int visibleElement;
	for(int i = 0; i < model->get_visible_point_nr_(); i++) {
		visibleElement = (model->get_visible_point_())[i] - 1; // Compensate for index translation from Matlab to C++

		cblas_dcopy(3, &Px[i*3], 1, &Px_large[visibleElement*3], 1);
		P1_large[visibleElement] = P1[i];
	}

	f.Px_large = Px_large;
	f.P1_large = P1_large;

	// Minimization steps
	vnl_lbfgs minimizer(f);	

	minimizer.set_max_function_evals(1000);					
	minimizer.set_x_tolerance(0.01);
	minimizer.set_f_tolerance(0.1);
	minimizer.set_g_tolerance(1);

	vnl_vector<double>* x_min_local = new vnl_vector<double>(shapeCoefLength + poseCoefLength);
	for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
		x_min_local->put(i, x_min[i]);
	}

	cout << "Starting minimizer." << endl;
	minimizer.minimize(*x_min_local);
	cout << "Minimizer completed." << endl;

	for(int i = 0; i < shapeCoefLength + poseCoefLength; i++) {
		x_min[i] = x_min_local->get(i);
	}

	free ((void*) x_min_local);
}


//=================================================computeMultiTransformationValueAndDerivative==//
// COMPUTEMULTITRANSFORMATIONVALUEANDDERIVATIVE computes the value and the
// derivative of transformations given a basis of transformation and weights
// associated to them.
// INPUTS: Basis is 7xL, where each column is a similarity transformation presented in log-euclidean space
//		   Theta is Lx1, and is weights associated to each basis
//		   poseMean is 4x4, representing the pose mean of these basis
// OUTPUT: deriv is 4x4xL
//		   value is 4x4
void Registration::computeMultiTransformationValueAndDerivative(double* theta, double* basis, int L, 
																double* poseMean, double* deriv, double* value)
{
	int temp_array_size;
	int onei = 1;
	double zero = 0.0;
	double one = 1.0;


	double* Tr = (double*)mkl_malloc(4 * 4 * L * sizeof(double), 64);
	double* transform_tmp = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* transform_tmp2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	
	// value = eye(4);
	//value = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* value2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 16; i++) value[i] = 0.0;
	for(int i = 0; i < 16; i+=5) value[i] = 1.0;
	

	for(int i = 0; i < L; i++) {
		
		//for(int p = 0; p < 16; p++) cout << basis[i+p] << " ";
		/*
		cout << L << endl;		
		cout << "basis" << endl;
		for(int p = 0; p < 7; p++) {
			for(int o = 0; o < 2; o++)
				cout << basis[i+ p*L + o] << "," << i+ p*L + o << "  ";
			cout << endl;
		}
		
		getchar();
		*/


		// Tr(:,:,l) = real(expm(theta(l)*getTransformationMatrixFromVector(basis(:,l))));
		//MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], L, transform_tmp);
		MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], 31, transform_tmp);

		/*
		cout << "transform_tmp" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << transform_tmp[p*4 + o] << " ";
			cout << endl;
		}
		getchar();
		*/

		//cout << "theta[i]: " << theta[i] << endl;
		//getchar();

		cblas_dscal(16, theta[i], transform_tmp, onei);
		
		/*
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << transform_tmp[p*4 + o] << "  ";
			cout << endl;
		}
		cout << "transform_tmp" << endl;
		getchar();
		*/

		// fails here
		MultiObjModelMKL::matrix_exponential(transform_tmp, transform_tmp2);		
		cblas_dcopy(16, transform_tmp2, 1, &Tr[i*16], 1);
		

		/*
		cout << "value" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << value[p*4 + o] << " ";
		cout << endl;
		}
		getchar();
		*/
		
		// value = value * Tr(:,:,l);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, value, 4, &Tr[i*16], 4, zero, value2, 4);
		/*
		cout << "value2" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << value2[p*4 + o] << " ";
		cout << endl;
		}
		getchar();
		*/
		cblas_dcopy(16, value2, 1, value, 1);
	}
	
	
	
	/* completely correct
	cout << "TR: " << endl;
	for(int i = 0; i < 4*2; i++) {
		for(int j = 0; j < 4; j++)
			cout << Tr[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/

	/*
	cout << "value" << endl;
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << value[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/

	// value = value * poseMean;	
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				4, 4, 4, one, value, 4, poseMean, 4, zero, value2, 4);
	cblas_dcopy(16, value2, 1, value, 1);
	
	/*
	cout << "value" << endl;
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << value[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/
	

	//deriv = (double*)mkl_malloc(4 * 4 * L * sizeof(double), 64);
	for(int i = 0; i < 4 * 4 * L; i++) deriv[i] = 0.0;
	
	
	double* deriv_intermd = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	for(int i = 0; i < 4 * 4; i++) deriv_intermd[i] = 0.0;

	
	for(int i = 0; i < L; i++) {

		// deriv(:,:,l) = eye(4);
		for(int x = 0; x < 4 * 4; x++) deriv[i*16 + x] = 0.0;
		for(int x = 0; x < 4 * 4; x+=5) deriv[i*16 + x] = 1.0;
		
		for(int p = 0; p < i; p++) {
			// deriv(:,:,l) = deriv(:,:,l) * Tr(:,:,p);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						4, 4, 4, one, &deriv[i*16], 4, &Tr[p*16], 4, zero, deriv_intermd, 4);
			cblas_dcopy(16, deriv_intermd, 1, &deriv[i*16], 1);
			
		}
		
		/*
		cout << "deriv-i" << endl;
		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/
		
		

		//MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], L, transform_tmp);
		MultiObjModelMKL::getTransformationMatrixFromVector(&basis[i], 31, transform_tmp);
 
		/* Correct
		cout << "transform_tmp" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << transform_tmp[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/

		// deriv(:,:,l) = deriv(:,:,l) * getTransformationMatrixFromVector(basis(:,l)) * Tr(:,:,l);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, &deriv[i*16], 4, transform_tmp, 4, zero, deriv_intermd, 4);
		
		/*
		cout << "deriv_intermd" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv_intermd[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, deriv_intermd, 4, &Tr[i*16], 4, zero, &deriv[i*16], 4);

		/*
		cout << "deriv-i2" << endl;
		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/
		

		cblas_dcopy(16, &deriv[i*16], 1, deriv_intermd, 1);
		for(int q = 1; q < L; q++) {
			// deriv(:,:,l) = deriv(:,:,l) * Tr(:,:,q);
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						4, 4, 4, one, deriv_intermd, 4, &Tr[q*16], 4, zero, &deriv[i*16], 4);

			/* correct
			cout << "deriv-i3" << endl;
			for(int p = 0; p < 4 * 2; p++) {
				for(int o = 0; o < 4; o++)
					cout << deriv[p*4 + o] << " ";
				cout << endl;

			}
			getchar();
			*/

			cblas_dcopy(16, &deriv[i*16], 1, deriv_intermd, 1);
			
		}

		// deriv(:,:,l) = deriv(:,:,l) * poseMean;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, one, deriv_intermd, 4, poseMean, 4, zero, &deriv[i*16], 4);

		/*
		cout << "poseMean" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << poseMean[p*4 + o] << " ";
			cout << endl;

		}
		getchar();

		cout << "deriv_intermd" << endl;
		for(int p = 0; p < 4; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv_intermd[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/

		/*
		cout << "deriv-final" << endl;
		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << deriv[p*4 + o] << " ";
			cout << endl;

		}
		getchar();
		*/
	}

	/*
	cout << "deriv" << endl;
	for(int i = 0; i < 4 * 2; i++) {
		for(int j = 0; j < 4; j++)
			cout << deriv[i*4 + j] << " ";
		cout << endl;

	}
	getchar();
	*/
	
	
	mkl_free_buffers();
	mkl_free(Tr);
	mkl_free(transform_tmp);
	mkl_free(transform_tmp2);
	mkl_free(value2);
	mkl_free(deriv_intermd);
	
	
}

//==========================================================computePointCorrValueAndDerivative===//
// COMPUTEPOINTCORRVALUEANDDERIVATIVE computes the value of the cost function
// and the derivative of it with respect to the weights given to pose
// variations.
// INPUT: BASIS is a 7xL
//		  POSECOEF is a Lx1
//        POSEMEAN is a 4x4
//        PX is a Nx3
//		  Y is Nx3
//        P1 is Nx1
// OUTPUT:DERIV is a Lx1
// 
void Registration::computePointCorrValueAndDerivative(double* poseCoef, int L, double* PX, double* P1,
													  double* poseMean, double* y, int N, double* basis,
													  int pose_modes_nr_, double* deriv)
{


	int temp_array_size;
	int onei = 1;
	double zero = 0.0;
	double one = 1.0;

	// [Tr TrDeriv] = computeMultiTransformationValueAndDerivative(poseCoef, basis(:,1:coefNr), poseMean);
	double* TrDeriv = (double*)mkl_malloc(4 * 4 * 2 * sizeof(double), 64);
	double* Tr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	
	
	// Need to copy the arrays
	double* basis_trim = (double*)mkl_malloc(7 * pose_modes_nr_ * sizeof(double), 64); // have to change 31 later
	temp_array_size = 7 * 2;
	int two = 2;
	/*
	dcopy(&temp_array_size, basis, &basis_cols, basis_trim, &two);
	dcopy(&temp_array_size, &basis[1], &basis_cols, &basis_trim[1], &two);
	cout << "trim: " << basis_trim[3] << endl;
	computeMultiTransformationValueAndDerivative(poseCoef, basis_trim, L, poseMean, TrDeriv, Tr);
	*/
	//cout << "compute1a" << endl;
	

	// Checking input to most inner funciton
	/*
	for(int p = 0; p < 2; p++) cout << poseCoef[p] << " ";
	cout << endl;
	cout << "poseCoef" << endl;
	getchar();


	for(int p = 0; p < 7; p++) {
		for(int o = 0; o < 2; o++) 
			cout << basis[p*31 + o] << " ";
		cout << endl;
	}
	cout << "basis" << endl;
	getchar();
	

	
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << poseMean[i*4 + j] << " ";
		cout << endl;

	}
	cout << "poseMean" << endl;
	getchar();
	*/
	computeMultiTransformationValueAndDerivative(poseCoef, basis, L, poseMean, TrDeriv, Tr);

	// Need to add in the calculation in order to return TrY's values
	
	//printMatrix(Tr, 4, 4, "Tr");
	//printMatrix(y, N, 3, "y");
	
	// TrY = bsxfun(@plus, y*Tr(1:3,1:3)', Tr(1:3,4)');
	double* TrY = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 3, one, y, 3, Tr, 4, zero, TrY, 3);
	
	//cout << "poedit1a" << endl;

	double* Nx1 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	for(int i = 0; i < N * 1; i++) Nx1[i] = 1.0;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 1, one, Nx1, 1, &Tr[3], 4, one, TrY, 3);
	
	//cout << "poedit2" << endl;

	/*
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << TrDeriv[i*4 + j] << " ";
		cout << endl;

	}
	cout << "TrDeriv" << endl;
	getchar();

	
	for(int i = 0; i < 4; i++) {
		for(int j = 0; j < 4; j++)
			cout << Tr[i*4 + j] << " ";
		cout << endl;

	}
	cout << "Tr" << endl;
	getchar();
	

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < 3; j++)
			cout << y[i*3 + j] << " ";
		cout << endl;

	}
	cout << "y " << y[0] << endl;
	getchar();
	
	

	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < 3; j++)
			cout << TrY[i*3 + j] << " ";
		cout << endl;

	}
	cout << "TrY " << TrY[0] << endl;
	getchar();
	*/
	
	
	
	// deriv = zeros(L, 1);
	//double* deriv = (double*)mkl_malloc(L * 1 * sizeof(double), 64);
	for(int i = 0; i < L; i++) deriv[i] = 0.0;

	double* TrDerivY = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	double* tmp1 = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	double* tmp2 = (double*)mkl_malloc(N * 3 * sizeof(double), 64);
	double* tmp3 = (double*)mkl_malloc(3 * 1 * sizeof(double), 64);
	double* sum1 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	double* sum2 = (double*)mkl_malloc(N * 1 * sizeof(double), 64);
	
	//cout << "compute5" << endl;
	
	for(int i = 0; i < L; i++) {
		// TrDerivY = bsxfun(@plus, y*TrDeriv(1:3,1:3,l)', TrDeriv(1:3,4,l)'); 
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
					N, 3, 3, one, y, 3, &TrDeriv[i*16], 4, zero, TrDerivY, 3);

		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
				N, 3, 1, one, Nx1, 1, &TrDeriv[i*16 + 3], 4, one, TrDerivY, 3);
		
		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++)
				cout << y[p*3 + o] << " ";
			cout << endl;

		}
		cout << "y " << y[0] << endl;
		getchar();
		

		for(int p = 0; p < 4 * 2; p++) {
			for(int o = 0; o < 4; o++)
				cout << TrDeriv[p*4 + o] << " ";
			cout << endl;

		}
		cout << "TrDeriv " << endl;
		getchar();
		
	
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++)
				cout << TrDerivY[p*3 + o] << " ";
			cout << endl;

		}
		cout << "TrDerivY " << TrDerivY[0] << endl;
		getchar();
		*/
		

		// deriv(l) = 2*sum(P1.*sum(TrY.*TrDerivY,2)) - 2*sum(sum(PX.*TrDerivY));
		temp_array_size = N * 3;
		vdMul(temp_array_size, TrY, TrDerivY, tmp1);

		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) {
				cout << tmp1[p*3 + o] << " ";
			}
			cout << endl;
		}
		cout << "tmp1" << endl;
		getchar();
		*/
		
		/*
		//optimize later po!
		temp_array_size = 3;
		for(int x = 0; x < N; x++) {
			sum1[x] = tmp1[x*3] + tmp1[x*3 + 1] + tmp1[x*3 + 2];
			//cout << sum1[x] << endl;
			
		}
		//getchar();
		
		
		temp_array_size = N * 1;
		dcopy(&temp_array_size, sum1, &onei, sum2, &onei);
		vdMul(temp_array_size, P1, sum2, sum1);
		for(int p = 0; p < N; p++) deriv[i] += 2 * sum1[p]; // optimize again po!
		*/

		cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
					1, 3, N, one, P1, 1, tmp1, 3, zero, tmp3, 3);
		deriv[i] = 2 * (tmp3[0] + tmp3[1] + tmp3[2]);
		

		
		/*
		for(int p = 0; p < N; p++) {
			cout << sum2[p] << endl;
		}
		cout << "sum2 " << endl;
		getchar();

	
		for(int p = 0; p < N; p++) {
			cout << P1[p] << endl;
		}
		cout << "P1 " << endl;					// P1 is different!!!
		getchar();

		for(int p = 0; p < N; p++) {
			cout << sum1[p] << endl;
		}
		cout << "sum1 " << endl;
		getchar();
		*/
		
		
		
		// correct up to here

		temp_array_size = N * 3;	
		vdMul(temp_array_size, PX, TrDerivY, tmp2);

		
		for(int p = 0; p < N * 3; p++)    /// sum of a vector
			deriv[i] -= 2 * tmp2[p];
	
		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) 
				cout << PX[p*3 + o] << " ";
			cout << endl;
		}
		cout << "PX " << endl;			// PX is different!
		getchar();
		*/

		/*
		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) 
				cout << TrDerivY[p*3 + o] << " ";
			cout << endl;
		}
		cout << "TrDerivY " << endl;		// TrDerivY is off in the 2nd it
		getchar();

		for(int p = 0; p < N; p++) {
			for(int o = 0; o < 3; o++) 
				cout << tmp2[p*3 + o] << " ";
			cout << endl;
		}
		cout << "tmp2 " << endl;
		getchar();
		

		cout << "derivhere: " << deriv[i] << endl;
		getchar();
		*/
	}
	
	/*
	for(int p = 0; p < 2; p++) {
		cout << deriv[p] << " ";
		
	}
	cout << endl;
	cout << "deriv " << endl;
	getchar();
	*/
	
	mkl_free_buffers();
	//mkl_thread_free_buffers();
	mkl_free(TrDeriv);
	mkl_free(Tr);
	mkl_free(basis_trim);
	mkl_free(TrY);
	mkl_free(Nx1);
	mkl_free(TrDerivY);
	mkl_free(tmp1);
	mkl_free(tmp2);
	mkl_free(sum1);
	mkl_free(sum2);
}

//==============================================multiObjectAtlasPoseCoefCostFunction=============//
// INPUT:  POSECOEF is a Lx1 vector
//		   PX is a Nx3
//		   GAMMA is the regularization matrix
// OUTPUT: VALUE is a scalar  --> not complete
//		   DERIV is a Lx1
void Registration::multiObjectAtlasPoseCoefCostFunction(double* poseCoef, int L, MultiObjModelMKL* model,
														double* shapeDeformed, double* PX,
														double* P1, double* Gamma,
														double* deriv)
{
	int onei = 1;

	//mkl_disable_fast_mm();

	double value = 0;
	double* derivTmp = (double*)mkl_malloc(L * 1 * sizeof(double), 64);
	double* derivTmp2 = (double*)mkl_malloc(L * 1 * sizeof(double), 64);
	for(int i = 0; i < L * 1; i++) derivTmp2[i] = 0.0;

	int temp_array_size = L*1;	
	
	//cout << "multiObjectAtlasPoseCoefCostFunction begin" << endl;
	for(int i = 0; i < model->get_obj_nr_(); i++) {
		/*
		[valueTmp derivTmp] = computePointCorrValueAndDerivative( 
				poseCoef, 
				PX(model.objInd(l,1):model.objInd(l,2),:),
				P1(model.objInd(l,1):model.objInd(l,2)), 
				model.poseMean(:,:,l), 
				shapeDeformed(model.objInd(l,1):model.objInd(l,2),:), 
				model.poseFeature(l*7-6:l*7,:));
		*/
		/*
		computePointCorrValueAndDerivative(double* poseCoef,
										   int L, 
										   double* PX, 
										   double* P1,
										   double* poseMean, 
										   double* y, 
										   int N, 
										   double* basis
										   double* deriv)
													  */
		/*
		cout << "basis" << endl;
		for(int p = 0; p < 7; p++) {
			for(int o = 0; o < 2; o++)
				cout << (model->get_pose_feature_())[i * model->get_pose_sigma_rows() + p*2 + o] << " ";
			cout << endl;
		}
		
		getchar();
		*/

		/*
		for(int p = 0; p < 1500; p++) {
			for(int o = 0; o < 1; o++)
				cout << P1[(model->get_obj_ind_())[i*2] - 1 + p + o] << " ";
			cout << endl;
		}
		*/
		
		/*
		cout << "PX " << (model->get_obj_ind_())[i*2+1] - (model->get_obj_ind_())[i*2] + 1 << endl;
		cout << "P1 " << P1[(model->get_obj_ind_())[i*2] - 1] << endl;
		getchar();
		*/

		Registration::computePointCorrValueAndDerivative(poseCoef,
														 L,
														 &PX[((model->get_obj_ind_())[i*2] - 1) * 3],
														 &P1[(model->get_obj_ind_())[i*2] - 1],
														 &(model->get_pose_mean_())[i*16],
														 &shapeDeformed[((model->get_obj_ind_())[i*2] - 1) * 3], 
														 (model->get_obj_ind_())[i*2+1] - (model->get_obj_ind_())[i*2] + 1,
														 &(model->get_pose_feature_())[i*7 * model->get_pose_modes_nr_()], 
														 model->get_pose_modes_nr_(),
														 derivTmp);
		//printMatrix(derivTmp, L, 1, "derivTmp");

		// &(model->get_pose_feature_())[i * model->get_pose_sigma_rows()],
		cblas_dcopy(L, deriv, 1, derivTmp2, 1);
		vdAdd(temp_array_size, derivTmp, derivTmp2, deriv);
	}
	
	/*
	cout << "outside deriv: " << deriv[0] << " " << deriv[1] << endl;
	getchar();
	*/

	// normalize the cost function by the number of points

	// deriv = deriv / model.objInd(end, 1);
	/*for(int i = 0; i < L; i++) {
		deriv[i] /= (model->get_obj_ind_())[(model->get_obj_nr_()-1)*2];
	}*/

	
	// deriv = deriv + Gamma*poseCoef;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				2, 1, 2, 1, Gamma, 2, poseCoef, 1, 1, deriv, 1);
	/* Older/Wrong version
	double* Gamma_x_poseCoef = (double*)mkl_malloc(2 * 1 * sizeof(double), 64);
	cblas_dcopy(L, poseCoef, 1, Gamma_x_poseCoef, 1);
	for(int i = 0; i < L; i++) Gamma_x_poseCoef[i] *= Gamma;
	for(int i = 0; i < L; i++) deriv[i] += Gamma_x_poseCoef[i];
	*/
	

	
	mkl_free_buffers();
	mkl_thread_free_buffers();

	//mkl_free(Gamma_x_poseCoef);
	mkl_free(derivTmp);
	mkl_free(derivTmp2);
}

//==============================================multiObjectAtlasPoseCoefCostFunction=============//
// INPUT:  POSECOEF is a Lx1 vector
//		   PX is a Nx3
//		   GAMMA is the regularization matrix
// OUTPUT: VALUE is a scalar  --> not complete
//		   DERIV is a Lx1
void Registration::multiObjectAtlasShapeCoefFunction(double* poseCoef, double* shapeCoef, int L, int* objInd,
													 double* shapeFeature, double* poseFeature,
													 double* poseDeformed, double* movedShape,
													 int obj_nr, double* poseMean, int shapeFeatureCols,
													 double* PX, double* P1, double* Gamma,
													 double* deriv)
{
	
	// Initialize some variables
	double* Tr = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* Tr2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* pose_feature_transform = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* pose_feature_transform2 = (double*)mkl_malloc(4 * 4 * sizeof(double), 64);
	double* part1 = (double*)mkl_malloc(1 * L * sizeof(double), 64);
	double* part2 = (double*)mkl_malloc(1 * L * sizeof(double), 64);
	double* part1_2 = (double*)mkl_malloc(L * sizeof(double), 64);

	double* ones = (double*)mkl_malloc(1 * 3 * sizeof(double), 64);
		for(int x = 0; x < 3; x++) ones[x] = 1.0;
	double* two = (double*)mkl_malloc(1 * sizeof(double), 64);
		two[0] = 2.0;

	for(int i = 0; i < L; i++)
		deriv[i] = 0.0;

	// Main Loop
	for(int i = 0; i < obj_nr; i++) {
		cout << "here1" << endl;

		int num_elements = (objInd[i*2 + 1] - objInd[i*2]) + 1;

		// Tr = eye(4)
		for(int x = 0; x < 16; x++) {
			Tr[x] = 0.0;
			Tr2[x] = 0.0;
		}
		for(int x = 0; x < 16; x += 5) {
			Tr[x] = 1.0;
			Tr2[x] = 1.0;
		}

		// Clear transformation matrix output
		for(int x = 0; x < 16; x++) {
			pose_feature_transform[x] = 0.0;
			pose_feature_transform2[x] = 0.0;
		}

		for(int j = 0; j < L; j++) {
			// getTransformationMatrixFromVector( model.poseFeature((l-1)*7+1: l*7, i) );
			MultiObjModelMKL::getTransformationMatrixFromVector( &poseFeature[i*7*31 + j], 31, pose_feature_transform );
			
			// Tr = Tr * real(expm(poseCoef(i)*getTransformationMatrixFromVector( ... )));
			for(int x = 0; x < 16; x++) 
				pose_feature_transform[x] *= poseCoef[j];
			MultiObjModelMKL::matrix_exponential(pose_feature_transform, pose_feature_transform2);

			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
						4, 4, 4, 1, Tr, 4, pose_feature_transform2, 4, 0, Tr2, 4);
			cblas_dcopy(16, Tr2, 1, Tr, 1);
		}
		cout << "here2" << endl;

		
		

		// Tr = real(Tr * model.poseMean(:,:,l));
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					4, 4, 4, 1, Tr2, 4, &poseMean[i * 16], 4, 0, Tr, 4);

		cout << "here3" << endl;


		// MTr = movedShape(model.objInd(l,1):model.objInd(l,2),:)*Tr(1:3,1:3);
		double* MTr = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					num_elements, 3, 3, 1, &movedShape[(objInd[i*2]-1) * 3], 3, Tr, 4, 0, MTr, 3);
		

		cout << "here4" << endl;

		/*
		for(int p = 0; p < num_elements; p++) {
			for(int o = 0; o < 3; o++) {
				cout << movedShape[objInd[i*2]-1 + p*3 + o] << " ";
			}
			cout << endl;
		}
		cout << "movedshape, l=" << i << endl;
		getchar();
		*/
		/*
		for(int p = 0; p < 2; p++) {
			for(int o = 0; o < 2; o++) {
				cout << objInd[p*2 + o] << " ";
			}
			cout << endl;
		}
		cout << "objInd" << i << endl;
		getchar();
		*/

		// PMTr = bsxfun(@times, MTr, P1(model.objInd(l,1):model.objInd(l,2)))';
		double* PMTr_temp = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					num_elements, 3, 1, 1, &P1[(int)objInd[i*2]-1], 1, ones, 3, 0, PMTr_temp, 3);

		cout << "here5" << endl;

		double* PMTr = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		vdMul(num_elements*3, MTr, PMTr_temp, PMTr);

		cout << "here6" << endl;

		// featurePart = model.shapeFeature(model.objInd(l,1)*3-2:model.objInd(l,2)*3,1:featureNr);
		// part1 = PMTr(:)'*featurePart;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					1, L, num_elements * 3, 1, PMTr, num_elements * 3, &shapeFeature[(objInd[i*2]-1)*3*shapeFeatureCols], shapeFeatureCols, 0, part1, L);
		

		cout << "here7" << endl;

		// PXTr = PX(model.objInd(l,1):model.objInd(l,2),:)*Tr(1:3,1:3);
		double* PXTr = (double*)mkl_malloc(num_elements * 3 * sizeof(double), 64);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					num_elements, 3, 3, 1, &PX[(objInd[i*2]-1)*3], 3, Tr, 4, 0, PXTr, 3);


		cout << "here8" << endl;


		// PXTr = PXTr';
		// part2 = PXTr(:)'*featurePart;
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					1, L, num_elements * 3, 1, PXTr, num_elements * 3, &shapeFeature[(objInd[i*2]-1)*3*shapeFeatureCols], shapeFeatureCols, 0, part2, L);
		
		cout << "here9" << endl;

		// shapeDeriv = shapeDeriv + 2*(part1-part2)';
		vdSub(L, part1, part2, part1_2);
		for(int x = 0; x < L; x++)
			deriv[x] += 2 * part1_2[x];

		// shapeDeriv = shapeDeriv + Gamma*shapeDeriv
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
					2, 1, 2, 1, Gamma, 2, deriv, 1, 1, deriv, 1);

		cout << "here10" << endl;

		// Free loop variables from memory
		//mkl_free_buffers();
		mkl_free(MTr);
		mkl_free(PMTr_temp);
		mkl_free(PMTr);
		mkl_free(PXTr);

	}

	cout << "Before free" << endl;
	// Free all local variables
	mkl_free_buffers();
	cout << "a" << endl;
	mkl_free(Tr);				// fails to free
	cout << "b" << endl;
	mkl_free(Tr2);
	cout << "c" << endl;
	mkl_free(pose_feature_transform);
	cout << "d" << endl;
	mkl_free(pose_feature_transform2);
	cout << "e" << endl;
	mkl_free(ones);
	cout << "f" << endl;
	mkl_free(part1);
	cout << "g" << endl;
	mkl_free(part2);
	cout << "h" << endl;
	mkl_free(part1_2);
	cout << "done." << endl;
	
}