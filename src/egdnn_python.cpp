#include <Python.h>
#include <arrayobject.h>
#include "egdnn.h"
using namespace EGDNN;

Egdnn * model;

static PyObject * _init(PyObject *self, PyObject *args)
{
	int input_N;
	int output_N;
	int populationSize;
	double learning_rate;
	double velocity_decay;
	double regularization_l2;
	double gradientClip;
	
	PyArg_ParseTuple(args, "iiidddd", &input_N, &output_N, &populationSize, &learning_rate, &velocity_decay, &regularization_l2, &gradientClip);
	
	model = new Egdnn(input_N, output_N, populationSize, learning_rate, velocity_decay, regularization_l2, gradientClip);
	
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject * _fit(PyObject *self, PyObject *args)
{
	int netId;
	std::vector<std::vector<double>> trainingSet;
	std::vector<std::vector<double>> trainingLabels;
	int iterNum;
	int batchSize;
	
	PyObject * trainingSet_obj;
	PyObject * trainingLabels_obj;
	PyArg_ParseTuple(args, "iOOii", &netId, &trainingSet_obj, &trainingLabels_obj, &iterNum, &batchSize);
	
	// parse trainingSet
	int trainingSet_dim0 = PyArray_DIMS(trainingSet_obj)[0];
	int trainingSet_dim1 = PyArray_DIMS(trainingSet_obj)[1];
	double * trainingSet_arr = static_cast<double *>(PyArray_DATA(trainingSet_obj));
	trainingSet.resize(trainingSet_dim0);
	for(int i = 0; i < trainingSet_dim0; i++)
	{
		trainingSet[i].resize(trainingSet_dim1);
		for(int j = 0; j < trainingSet_dim1; j++)
		{
			trainingSet[i][j] = *(trainingSet_arr + i * trainingSet_dim1 + j);
		}
	}
	
	// parse trainingLabels
	int trainingLabels_dim0 = PyArray_DIMS(trainingLabels_obj)[0];
	int trainingLabels_dim1 = PyArray_DIMS(trainingLabels_obj)[1];
	double * trainingLabels_arr = static_cast<double *>(PyArray_DATA(trainingLabels_obj));
	trainingLabels.resize(trainingLabels_dim0);
	for(int i = 0; i < trainingLabels_dim0; i++)
	{
		trainingLabels[i].resize(trainingLabels_dim1);
		for(int j = 0; j < trainingLabels_dim1; j++)
		{
			trainingLabels[i][j] = *(trainingLabels_arr + i * trainingLabels_dim1 + j);
		}
	}
	
	model->fit(netId, trainingSet, trainingLabels, iterNum, batchSize);
	
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject * _predict(PyObject *self, PyObject *args)
{
	int netId;
	std::vector<double> data;
	std::vector<double> prediction;
	
	PyObject * data_obj;
	PyArg_ParseTuple(args, "iO", &netId, &data_obj);
	
	// parse data
	int data_dim = PyArray_DIMS(data_obj)[0];
	double * data_arr = static_cast<double *>(PyArray_DATA(data_obj));
	data.resize(data_dim);
	for(int i = 0; i < data_dim; i++)
	{
		data[i] = data_arr[i];
	}
	
	// get prediction
	prediction = model->predict(netId, data);
	
	// construct ndarray
	npy_intp prediction_dims[] = {prediction.size()};
	PyObject * prediction_obj = PyArray_SimpleNew(1, prediction_dims, NPY_FLOAT64);	
	double * prediction_arr = static_cast<double *>(PyArray_DATA(prediction_obj));
	for(int i = 0; i < prediction.size(); i++)
	{
		prediction_arr[i] = prediction[i];
	}
	
	return prediction_obj;
}

static PyObject * _predict_batch(PyObject *self, PyObject *args)
{
	int netId;
	std::vector<std::vector<double>> data;
	
	PyObject * data_obj;
	PyArg_ParseTuple(args, "iO", &netId, &data_obj);
	
	// parse data
	int data_dim0 = PyArray_DIMS(data_obj)[0];
	int data_dim1 = PyArray_DIMS(data_obj)[1];
	double * data_arr = static_cast<double *>(PyArray_DATA(data_obj));
	data.resize(data_dim0);
	for(int i = 0; i < data_dim0; i++)
	{
		data[i].resize(data_dim1);
		for(int j = 0; j < data_dim1; j++)
		{
			data[i][j] = *(data_arr + i * data_dim1 + j);
		}
	}
	
	// construct ndarray
	npy_intp prediction_dims[] = {data_dim0, model->output_N};
	PyObject * prediction_obj = PyArray_SimpleNew(2, prediction_dims, NPY_FLOAT64);	
	double * prediction_arr = static_cast<double *>(PyArray_DATA(prediction_obj));
	for(int i = 0; i < data_dim0; i++)
	{
		// get prediction
		std::vector<double> prediction = model->predict(netId, data[i]);
		for(int j = 0; j < model->output_N; j++)
		{
			*(prediction_arr + i * model->output_N + j) = prediction[j];
		}
	}
	
	return prediction_obj;
}

static PyObject * _test(PyObject *self, PyObject *args)
{
	int netId;
	std::vector<std::vector<double>> testSet;
	std::vector<std::vector<double>> testLabels;
	double accuracy;
	
	PyObject * testSet_obj;
	PyObject * testLabels_obj;
	PyArg_ParseTuple(args, "iOO", &netId, &testSet_obj, &testLabels_obj);
	
	// parse testSet
	int testSet_dim0 = PyArray_DIMS(testSet_obj)[0];
	int testSet_dim1 = PyArray_DIMS(testSet_obj)[1];
	double * testSet_arr = static_cast<double *>(PyArray_DATA(testSet_obj));
	testSet.resize(testSet_dim0);
	for(int i = 0; i < testSet_dim0; i++)
	{
		testSet[i].resize(testSet_dim1);
		for(int j = 0; j < testSet_dim1; j++)
		{
			testSet[i][j] = *(testSet_arr + i * testSet_dim1 + j);
		}
	}
	
	// parse testLabels
	int testLabels_dim0 = PyArray_DIMS(testLabels_obj)[0];
	int testLabels_dim1 = PyArray_DIMS(testLabels_obj)[1];
	double * testLabels_arr = static_cast<double *>(PyArray_DATA(testLabels_obj));
	testLabels.resize(testLabels_dim0);
	for(int i = 0; i < testLabels_dim0; i++)
	{
		testLabels[i].resize(testLabels_dim1);
		for(int j = 0; j < testLabels_dim1; j++)
		{
			testLabels[i][j] = *(testLabels_arr + i * testLabels_dim1 + j);
		}
	}
	
	accuracy = model->test(netId, testSet, testLabels);
	
	return PyFloat_FromDouble(accuracy);
}

static PyObject * _evolution(PyObject *self, PyObject *args)
{
	int bestNetId;
	
	PyArg_ParseTuple(args, "i", &bestNetId);
	
	model->evolution(bestNetId);
	
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject * _kbhit(PyObject *self, PyObject *args)
{
	return PyBool_FromLong(kbhit());
}

static PyObject * _display(PyObject *self, PyObject *args)
{
	model->display();
	
	Py_INCREF(Py_None);
	return Py_None;
}

static PyMethodDef Methods[] = 
{
	{
		"init",
		_init,
		METH_VARARGS,
		""
	},
	{
		"fit",
		_fit,
		METH_VARARGS,
		""
	},
	{
		"predict",
		_predict,
		METH_VARARGS,
		""
	},
	{
		"predict_batch",
		_predict_batch,
		METH_VARARGS,
		""
	},
	{
		"test",
		_test,
		METH_VARARGS,
		""
	},
	{
		"evolution",
		_evolution,
		METH_VARARGS,
		""
	},
	{
		"display",
		_display,
		METH_VARARGS,
		""
	},
	{
		"kbhit",
		_kbhit,
		METH_VARARGS,
		""
	},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef egdnn_python =
{
	PyModuleDef_HEAD_INIT,
	"egdnn_python",
	NULL,
	-1,
	Methods
};

PyMODINIT_FUNC PyInit_egdnn_python(void)
{
	import_array();
	return PyModule_Create(&egdnn_python);
}
