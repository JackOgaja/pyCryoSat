/*
Extension module for reading CryoSat data sets
author: Jack Ogaja, jack_ogaja@brown.edu
*/

#include <Python.h>
#include "pyCryoSatIO.h"

static char module_docstring[] =
    "An extension for CryoSat2 Level 2 data IO";
static char l2Iarray_docstring[] =
    "function to read CryoSat2 L2 data into numpy arrays";

static PyObject *pyCryoSatIO_l2Iarray(PyObject *self, PyObject *args);
static PyObject *arrayError; /*-- unique exception object--*/ 

//die Hauptfunktion
static PyObject *pyCryoSatIO_l2Iarray(PyObject *self, PyObject *args)
{
    int narg;
    const char *fname;

    /*-- Parse the input tuple --*/
    if (!PyArg_ParseTuple(args, "is", &narg, &fname))
        return NULL;

    char *fileName = (char *)fname;
    BASELINE fbase = (BASELINE) narg;

    /*-- Call the wrapped function --*/
    //int cn = main(narg=2, &fileName);
    int cn = csarray(fileName, fbase);

    /*-- raise an exception if necessary --*/
   if (cn < 0) {
       PyErr_SetString(arrayError,
                    "There is a problem with the l2Iarray function call");
        return NULL;
    }

     return PyLong_FromLong(cn);

} /* die Hauptfunktion */ 

static PyMethodDef pyCryoSatIO_methods[] = {
    {"l2Iarray", pyCryoSatIO_l2Iarray, METH_VARARGS, l2Iarray_docstring},
    {NULL, NULL, 0, NULL}
}; 

PyMODINIT_FUNC PyInit_pyCryoSatIO(void)
{
    PyObject *module;
    static struct PyModuleDef pyCryoSatIO_module = {
        PyModuleDef_HEAD_INIT,
        "pyCryoSatIO",
        module_docstring,
        -1,
        pyCryoSatIO_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    module = PyModule_Create( &pyCryoSatIO_module );
    if (!module) return NULL;

    /*-- create a unique exception object --*/
    arrayError = PyErr_NewException("pyCryoSatIO.error", NULL, NULL);
    Py_INCREF(arrayError);
    PyModule_AddObject(module, "error", arrayError);

    return module;
}

