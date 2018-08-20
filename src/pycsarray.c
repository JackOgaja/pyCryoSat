/* Copyright (C) 2018 Jack Ogaja.
*
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
*
*  The above copyright notice and this permission notice shall be included in all
*  copies or substantial portions of the Software.
*
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*  SOFTWARE.
*/

/**********
 + Python C-extension module for reading European Space Agency's 
   CryoSat2 satellite data 
 + This extention uses I/O libraries prepared by the software team at
   Mullard Space Science Laboratory(MSSL), UCL London.

  Jack Ogaja, 
  Brown University,
  20180620
  See LICENSE.md for copyright notice
***********/

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

/*-- create a numpy array from a C -vector or -matrix --*/
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

// the main function
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
    //printf( "There are %ld records in the input file %s\n",
    //       num_recs, fileName );

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
      case Day:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Day;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Sec:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->uj_Sec;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Micsec:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->uj_Micsec;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case USO_Correction_factor:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_USO_Corr;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Mode_id:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_Mode_ID;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Src_Seq_Counter:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_SSC;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Instrument_config:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Inst_config;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Rec_Counter:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Rec_Count;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Lat:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Lat;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Lon:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Lon;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Altitude:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Alt;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Altitude_rate:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Alt_rate;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Satellite_velocity: // matrix[3][num_recs]
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
      case Real_beam: // matrix[3][num_recs]
           {
             dims[0] = 3;
             dims[1] = num_recs;
             strides[0] = sizeof(int32_t);
             strides[1] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->aj_Real_beam;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(2, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(2, dims, fp, arrayStruct); 
             break;
           }
      case Baseline: // matrix[3][num_recs]
           {
             dims[0] = 3;
             dims[1] = num_recs;
             strides[0] = sizeof(int32_t);
             strides[1] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->aj_Baseline;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(2, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(2, dims, fp, arrayStruct); 
             break;
           }
      case Star_Tracker_id:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->i_ST_ID;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Roll_angle:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Roll_angle;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Pitch_angle:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Pitch_angle;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Yaw_angle:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Yaw_angle;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Level2_MCD:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_L2_MCD;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Height:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Height;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Height_2:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Height_2;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Height_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Height_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Sigma_0:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Sig0;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Sigma_0_2:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Sig0_2;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Sigma_0_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Sig0_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Significant_Wave_Height:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SWH;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Peakiness:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Peakiness;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_range_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_range_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_range_correction_2:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_range_C_2;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_range_correction_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_range_C_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_sig0_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_sig0_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_sig0_correction_2:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_sig0_C_2;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_sig0_correction_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_sig0_C_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_quality_1:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_Quality_1;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_quality_2:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_Quality_2;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_quality_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_Quality_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_4:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_4;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_5:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_5;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_6:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_6;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_7:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_7;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_8:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_8;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_9:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_9;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_10:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_10;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_11:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_11;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_12:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_12;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_13:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_13;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_14:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_14;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_15:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_15;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_16:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_16;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_17:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_17;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_18:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_18;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_19:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_19;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_20:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_20;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_21:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_21;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_22:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_22;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracked_23:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Retrk_23;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Power_echo_shape:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_echo_shape;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Beam_behaviour_param: // matrix[3][num_recs]
           {
             dims[0] = 50;
             dims[1] = num_recs;
             strides[0] = sizeof(int16_t);
             strides[1] = sizeof(L2IData);
             int16_t *arrayStruct = &(&arrayPtr[0])->ai_bb_param;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(2, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(2, dims, fp, arrayStruct); 
             break;
           }
      case Cross_track_angle:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_XTrack_angle;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Cross_track_angle_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_XTrack_angle_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Coherence:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Coherence;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
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
      case Interpolated_sea_surface_h_anom:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SSHA_interp;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ocean_height_interpolation_error:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_interp_err;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Interpolated_forward_records:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_interp_cnt_fwd;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Interpolated_backward_records:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_interp_cnt_bkwd;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Time_interpolated_fwd_recs:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_interp_time_fwd;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Time_interpolated_bkwd_recs:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_interp_time_bkwd;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Interpolation_error_flag:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_interp_error_F;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Measurement_mode:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Meas_Mode;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Quality_flags:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Quality_F;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Retracker_flags:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Retracker_F;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Height_status:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Ht_status;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Freeboard_status:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Freeb_status;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Average_echoes:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_n_avg;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Wind_speed:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint16_t *arrayStruct = &(&arrayPtr[0])->ui_Wind_speed;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ice_concentration:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Ice_conc;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Snow_depth:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Snow_depth;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Snow_density:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Snow_density;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Discriminator:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Discriminator;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_1:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_1;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_2:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_2;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_3:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_3;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_4:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_4;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_5:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_5;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_6:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_7;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_7:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_7;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_8:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_8;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_9:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_9;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SARin_discriminator_10:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SARin_disc_10;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Discriminator_flags:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Discrim_F;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Attitude:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Attitude;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Azimuth:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Azimuth;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Slope_doppler_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Slope_Doppler_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Satellite_latitude:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Lat_sat;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Satellite_longitude:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Lon_sat;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ambiguity:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Ambiguity;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case MSS_model:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_MSS_mod;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Geoid_model:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Geoid_mod;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case ODLE_model:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_ODLE_mod;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case DEM_elevation:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_DEM_elev;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case DEM_id:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_DEM_id;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Dry_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Dry_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Wet_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Wet_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case IB_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_IB_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case DAC_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_DAC_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ionospheric_GIM_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Iono_GIM;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ionospheric_model_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Iono_mod;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ocean_tide:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_H_OT;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case LPE_Ocean_tide:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_H_LPEOT;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Ocean_loading_tide:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_H_OLT;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Solid_earth_tide:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_H_SET;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Geocentric_polar_tide:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_H_GPT;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Surface_type:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Surf_type;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Correction_status:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Corr_status;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Correction_error:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             uint32_t *arrayStruct = &(&arrayPtr[0])->uj_Corr_error;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case SS_Bias_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_SSB;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Doppler_rc:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Dopp_rc;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case TR_instrument_rc:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_TR_inst_rc;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case R_instrument_rc:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_R_inst_rc;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case TR_instrument_gain_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_TR_inst_gain_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case R_instrument_gain_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_R_inst_gain_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Internal_phase_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Int_phase_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case External_phase_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Ext_phase_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Noise_power_measurement:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Noise_pwr;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      case Phase_slope_correction:
           {
             dims[0] = num_recs;
             strides[0] = sizeof(L2IData);
             int32_t *arrayStruct = &(&arrayPtr[0])->j_Phase_slope_C;
             int8_t itemsize = fieldSize(field);
             int typeNum = NPY_INT;
             fp = getProperties(1, typeNum, itemsize, strides);
             arrayObj = array_CRYOSAT(1, dims, fp, arrayStruct); 
             break;
           }
      default:
           PyErr_SetString(PyExc_ValueError,
                           "Data field unknown! \n \
                            Select between 1 and 131");
           return NULL;
    }

    return PyArray_Return(arrayObj);
      
} /* the main function */ 

