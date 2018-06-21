/*

 + Python c-extension module for reading CryoSat2 satellite data 
 + This extention uses I/O libraries prepared by the team at
   Mullard Space Science Laboratory(MSSL), UCL London.

  Jack Ogaja, 
  Brown University,
  jack_ogaja@brown.edu
  20180620
  See LICENSE.md for copyright information

*/

#include <Python.h>

/*-- this definition is necessary for NUMPY API calls in a separate module --*/
#define PY_ARRAY_UNIQUE_SYMBOL pycryo_ARRAY_API

#include "csarray.h"

/*-- local functions prototypes --*/
static PyObject *csarray_l2Iarray(PyObject *self, PyObject *args);
static PyObject *cryosatArrayError; // unique exception object 
static PyArrayObject* array_CRYOSAT2(npy_intp* dims, 
                          field_properties f_p, void* arrayStruct); 

/*-- python doc-strings --*/
static char module_docstring[] =
    "A python c-extension for CryoSat2 Level 2i data IO";
static char l2Iarray_docstring[] =
    "function to read CryoSat2 L2i data into numpy arrays";

/*-- module methods --*/
static PyMethodDef csarray_methods[] = {
    {"l2Iarray", csarray_l2Iarray, METH_VARARGS, l2Iarray_docstring},
    {NULL, NULL, 0, NULL}
}; 

/*-- module initialization and numpy functionality --*/
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

    /* Load `numpy` functionality. */
    import_array();

    return module;
}

/*-- create a numpy array from satellite data fields--*/
static PyArrayObject*
array_CRYOSAT2(npy_intp* dims, field_properties f_p,
                                  void* arrayStruct)
{
      f_p.strides[0] = sizeof(L2IData);
      PyArray_Descr *descr = PyArray_DescrFromType(f_p.typenum);
      if (descr == NULL)
      {
          return NULL;
      }

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
                                      1, dims,
                                      f_p.strides, arrayStruct,
                                      NPY_ARRAY_CARRAY, NULL);

      return arrayRet;
} /* array_CRYOSAT2 */

//die Hauptfunktion
static PyObject *csarray_l2Iarray(PyObject *self, PyObject *args)
{
    int narg, fieldNum;
    npy_intp dims[NPY_MAXDIMS]; //shape of array
    long int num_recs; //total number of records in a file
    const char *fname;
    PyArrayObject  *arrayObj = NULL;
    t_cs_filehandle fHandle = NULL; //pointer to file info structure
    field_properties fp;

    /*-- Parse the input tuple --*/
    if (!PyArg_ParseTuple(args, "sii", &fname, &narg, &fieldNum))
        return NULL;

    char *fileName = (char *)fname;
    BASELINE fbase = (BASELINE) narg;
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

    dims[0] = num_recs; // number of elements in the dimension
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
           assert(!"Field unknown!"); return NULL;
      case Interpolated_Ocean_Height:
           {
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Ocean_ht;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(typeNum, itemsize);
             arrayObj = array_CRYOSAT2(dims, fp, arrayStruct); 
             break;
           }
      case Freeboard:
           {
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Freeboard;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(typeNum, itemsize);
             arrayObj = array_CRYOSAT2(dims, fp, arrayStruct); 
             break;
           }
      case Surface_Height_Anomaly:
           {
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SHA;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(typeNum, itemsize);
             arrayObj = array_CRYOSAT2(dims, fp, arrayStruct); 
             break;
           }
      default:
           assert(!"Field unknown!"); return NULL;
    }

    return PyArray_Return(arrayObj);
      
} /* die Hauptfunktion */ 

