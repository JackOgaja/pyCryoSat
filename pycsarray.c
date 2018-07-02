/*

 + Python C-extension module for reading European Space Agency's 
   CryoSat2 satellite data 
 + This extention uses I/O libraries prepared by the team at
   Mullard Space Science Laboratory(MSSL), UCL London.

  Jack Ogaja, 
  Brown University,
  jack_ogaja@brown.edu
  20180620
  See LICENSE.md for copyright notice

*/

#include <Python.h>

/*-- this definition is necessary for NUMPY API calls in a separate module --*/
#define PY_ARRAY_UNIQUE_SYMBOL pycryo_ARRAY_API

#include "csarray.h"

/*-- local functions prototypes --*/
static PyObject *csarray_l2Iarray(PyObject *self, PyObject *args);
static PyObject *cryosatArrayError; // unique exception object 
static PyArrayObject* array_CRYOSAT(int nd, npy_intp* dims, 
                          field_properties f_p, void* arrayStruct); 

/*-- python doc-strings --*/
static char module_docstring[] =
    "A python c-extension for CryoSat2 Level 2i data I/O";
static char l2Iarray_docstring[] =
    "function to read CryoSat2 L2i data into numpy arrays";

/*-- module methods --*/
static PyMethodDef csarray_methods[] = {
    {"l2Iarray", csarray_l2Iarray, METH_VARARGS, l2Iarray_docstring},
    {NULL, NULL, 0, NULL}
}; 

/*-- module initialization with numpy functionality --*/
PyMODINIT_FUNC PyInit_csarray(void)
{
    PyObject *module;
    static struct PyModuleDef csarray_module = {
        PyModuleDef_HEAD_INIT,
        "csarray",
        module_docstring,
        -1,
        csarray_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    module = PyModule_Create( &csarray_module );
    if (!module) return NULL;

    /*-- create a unique exception object --*/
    cryosatArrayError = PyErr_NewException("csarray.error", NULL, NULL);
    Py_INCREF(cryosatArrayError);
    PyModule_AddObject(module, "error", cryosatArrayError);

    /* Load numpy functionality. */
    import_array();

    return module;
}

/*-- create a numpy array from a vector or matrix--*/
static PyArrayObject*
array_CRYOSAT(int nd, npy_intp* dims, field_properties f_p,
                                  void* arrayStruct)
{
      PyArray_Descr *descr = PyArray_DescrFromType(f_p.typenum);
      if (descr == NULL) return NULL; 

      if (PyDataType_ISUNSIZED(descr))
      {
          if (f_p.typesize < 1)
          {
              PyErr_SetString(PyExc_ValueError,
                              "data type must provide an itemsize");
              Py_DECREF(descr);
              return NULL;
          }
          PyArray_DESCR_REPLACE(descr);
          descr->elsize = f_p.typesize;
      }

      PyArrayObject* arrayRet = (PyArrayObject *)
                 PyArray_NewFromDescr(&PyArray_Type, descr,
                                      nd, dims,
                                      f_p.strides, arrayStruct,
                                      NPY_ARRAY_CARRAY, NULL);

      return arrayRet;
} /* array_CRYOSAT */

//die Hauptfunktion
static PyObject *csarray_l2Iarray(PyObject *self, PyObject *args)
{
    int base, fieldNum;
    npy_intp dims[NPY_MAXDIMS]; //shape of array
    npy_intp strides[NPY_MAXDIMS]; 
    long int num_recs; //total number of records in a file
    const char *fname;
    PyArrayObject  *arrayObj = NULL;
    t_cs_filehandle fHandle = NULL; //pointer to file info structure
    field_properties fp;

//    int32_t **matcs = NULL;
    int32_t **matcs;
//    L2IData **matcs;

    /*-- Parse the input tuple --*/
    if (!PyArg_ParseTuple(args, "sii", &fname, &base, &fieldNum))
        return NULL;

    char *fileName = (char *)fname;
    BASELINE fbase = (BASELINE) base;
    FIELDS field = (FIELDS) fieldNum;

    /*-- get the file pointer --*/
    // MSSL I/O librarry
    fHandle = ptCSGetFileHandle( fileName, fbase );

    /*--open the data file for reading--*/
    num_recs = howManyRecs(fHandle, fbase);

    /*--check whether the file is correctly read--*/
    if( num_recs == -1 )
    {
      printf( "The number of records in the file cannot be determined.\n" );
      return NULL;
    }
    printf( "There are %ld records in the input file %s\n",
           num_recs, fileName );

    L2IData* arrayPtr = csarray(fHandle, num_recs);
    // MSSL I/O librarry
    vCSFreeFileHandle( fHandle );

    /*-- raise an exception if necessary --*/
    if (!arrayPtr) 
    {
      PyErr_SetString(cryosatArrayError,
                  "There is no data read from the input file");
      return NULL;
    }

    switch(field)
    {
      case Field_Unknown:
           PyErr_SetString(PyExc_ValueError,
                           "Unknown field selected as 0!");
           return NULL;
      case Satellite_velocity:
           {
             dims[0] = 3;
             dims[1] = num_recs;
             strides[0] = sizeof(int32_t);
             strides[1] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->aj_Sat_velocity;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(2, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(2, dims, fp, arrayStruct); 
             break;
           }
      case Interpolated_Ocean_Height:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Ocean_ht;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Freeboard:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Freeboard;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Surface_Height_Anomaly:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SHA;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      default:
           PyErr_SetString(PyExc_ValueError,
                           "Data field unknown! \n \
                            Select between 1 and 133");
           return NULL;
    }

    return PyArray_Return(arrayObj);
      
} /* die Hauptfunktion */ 

