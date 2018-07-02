/*--functions for reading ESA's CryoSat2 data sets--*/

/*==============
These functions use I/O libraries developed by UCL/MSSL, London
I/O library author: (C) CryoSat2 Software Team, 
                   Mullard Space Science Lab, UCL, London
I/O library version: 2.3
================*/

/*
Jack Ogaja, 
Brown University,
jack_ogaja@brown.edu
20180620
See LICENSE.md for copyright notice
*/

/*-- 
  Make NUMPY API calls available in this module 
--*/
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pycryo_ARRAY_API

#include "csarray.h"

#define FP field_properties

/*-- determine the sizes of different fields --*/
uint8_t
fieldSize( FIELDS field )
{
 switch(field)
    {
      case Field_Unknown:             return 0; 
      case Satellite_velocity:        return SAT_VELOCITY_SIZE; 
      case Interpolated_Ocean_Height: return OCEAN_HT_SIZE; 
      case Freeboard:                 return FREEBOARD_SIZE; 
      case Surface_Height_Anomaly:    return SHA_SIZE; 
      default: 
           assert(!"Field unknown!"); return 0;
    }
}

/*-- obtain fields properties --*/
FP
getProperties(int nd, int typenum, int typesize, npy_intp* ss)
{
    int j;
    FP fp;

    fp.typenum = typenum;
    fp.typesize = typesize;
    for (j = 0; j < nd; ++j) {
        fp.strides[j] = ss[j];
    }

    return fp;
} /* getProperties */

/*-- determine the total number of records in each field --*/
long int 
howManyRecs(t_cs_filehandle fH, BASELINE fBaseline)
{
    long int  n_records;

    // MSSL I/O librarry
    iTestEndian(); // test endianess

    if( fH != NULL )
    {
        // MSSL I/O librarry
        n_records = jCSNumRecordsInDataset( fH, 0 );
        return n_records;
    }
    else
    {
        printf( "Unable to open file.\n" );
        return EXIT_FAILURE;
    }

} /*howManyRecs*/

/*-- obtain the field data --*/
L2IData* 
csarray( t_cs_filehandle fH, long int n_records ) 
{

    L2IData*  fDataL2 = NULL;
    // MSSL I/O librarry
    fDataL2 = ptCSGetL2I( fH, 0, n_records, 0, NULL );

    if( fDataL2 != NULL )
    {
       /*
       Debug
       */
       printf( "L2 day [0] = %"PRId32"\n", fDataL2[ 0 ].j_Day );
       printf( "L2 day [1] = %"PRId32"\n", fDataL2[ 1 ].j_Day );
       printf( "S1 = %d\n", S1 );
       printf( "S2 = %d\n", S2 );
       printf( "SIZE_LIST_1 = %d\n", SIZE_LIST_1 );
       printf( "SIZE_LIST_2 = %d\n", SIZE_LIST_2 );

       return fDataL2;

    }
    else
    {
        printf( "Unable to read from file.\n" );
        return NULL;
    }

} /* csarray */

