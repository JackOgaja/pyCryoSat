
#ifndef _CSARRAY_H_
#define _CSARRAY_H_

/*
The module interface
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include <numpy/arrayobject.h>

// MSSL I/O INTERFACE
#include "ptCSGetFileHandle.h"
#include "vCSFreeFileHandle.h"
#include "ptCSGetL2I.h"
#include "jCSNumRecordsInDataset.h"
#include "jIOFunctions.h"
#include "CS_Defines.h"

#define OCEAN_HT_SIZE (4)
#define FREEBOARD_SIZE (4)
#define SHA_SIZE (4)

// New data types
typedef struct _field_properties {
   /*
    some unique properties of the data fields 
   */
   int typenum;
   /* the array data-type */
   int typesize;
   /* the field size in bytes */
   npy_intp strides[NPY_MAXDIMS];
   /* array strides */
} field_properties;

typedef struct _arrayStruct {
    /*
     construct the field arrays 
    */
    void* arrayData;
    /* the field array data */
    field_properties fp;
    /* the fields properties */
} arrayStruct;

/*
Select different data fields
*/
typedef enum _FIELDS
{
   /*-- add as many fields as necessary --*/
   Field_Unknown,
   Interpolated_Ocean_Height,
   Freeboard,
   Surface_Height_Anomaly
} FIELDS;

// functions prototypes
uint8_t fieldSize( FIELDS field ); 
long int howManyRecs( t_cs_filehandle fH, BASELINE fBaseline );
field_properties getProperties( int n, int s );
L2IData* csarray( t_cs_filehandle fH, long int n_records );

#endif // #ifndef _CSARRAY_H_

