#include <python3.5m/Python.h>
#include "egdnn.h"
using namespace EGDNN;

Egdnn * model;

static PyObject * _init(PyObject *self, PyObject *args)
{
	model = new Egdnn(1, 1, 1, 1, 1, 1, 1);
	return PyLong_FromLong(0);
}

static PyObject * _set(PyObject *self, PyObject *args)
{
	int val;	
	PyArg_ParseTuple(args, "i", &val);
	model->set(val);	
	return PyLong_FromLong(0);
}

static PyObject * _get(PyObject *self, PyObject *args)
{
	int val;
	val = model->get();
	return PyLong_FromLong(val);
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
		"get",
		_get,
		METH_VARARGS,
		""
	},
	{
		"set",
		_set,
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
	PyObject *m;
	m = PyModule_Create(&egdnn_python);
	if(m == NULL)
	{
		return NULL;
	}
	return m;
}
