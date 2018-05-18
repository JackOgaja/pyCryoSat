/*
A module for reading CryoSat2 data sets
author: Jack Ogaja, jack_ogaja@brown.edu
*/

#include <Python.h>
#include "pyCryoSatIO.h"

static char module_docstring[] =
    "An interface for reading CryoSat data";
static char example_docstring[] =
    "An example function to dump CryoSat data";

/*-- unique exception object--*/
static PyObject *exampleError;

//die Hauptfunktion
static PyObject *pyCryoSatIO_example(PyObject *self, PyObject *args)
{
    int narg;
    const char *fname;

    /*-- Parse the input tuple --*/
    if (!PyArg_ParseTuple(args, "is", &narg, &fname))
        return NULL;

    char *fileName = (char *)fname;

    /*-- Call the wrapped function --*/
    int cn = main(narg=2, &fileName);

    /*-- raise an exception if necessary --*/
   if (cn < 0) {
       PyErr_SetString(exampleError,
                    "There is a problem with the main function call");
        return NULL;
    }

     return PyLong_FromLong(cn);

} /* die Hauptfunktion */ 

static PyMethodDef pyCryoSatIO_methods[] = {
    {"example", pyCryoSatIO_example, METH_VARARGS, example_docstring},
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
    exampleError = PyErr_NewException("pyCryoSatIO.error", NULL, NULL);
    Py_INCREF(exampleError);
    PyModule_AddObject(module, "error", exampleError);

    return module;
}

