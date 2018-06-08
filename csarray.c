/*--functions for reading CryoSat2 data sets--*/

/*==============
These functions use IO libraries developed by MSSL, UCL, London
IO library author: Copyright CryoSat2 Software Team, 
                   Mullard Space Science Lab, UCL, London
IO library version: 2.3
================*/

/*--author: Jack Ogaja, jack_ogaja@brown.edu--*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include "ptCSGetFileHandle.h"
#include "vCSFreeFileHandle.h"
#include "ptCSGetL2I.h"
#include "jCSNumRecordsInDataset.h"
#include "jIOFunctions.h"
#include "CS_Defines.h"

#include "pyCryoSatIO.h"

int csarray( char* inFile,
                  BASELINE fBaseline )
{

    L2IData*  fDataL2 = NULL;
    t_cs_filehandle fHandle = NULL;
    long int  n_records;

    iTestEndian(); // test endianess

    fHandle = ptCSGetFileHandle( inFile, fBaseline );

    if( fHandle != NULL )
    {
        n_records = jCSNumRecordsInDataset( fHandle, 0 );

        if( n_records == -1 )
        {
            printf( "Unable to get number of records in file.\n" );
            return EXIT_FAILURE;
        }

        printf( "There are %ld records in the file %s\n",
                n_records, inFile );

        fDataL2 = ptCSGetL2I( fHandle,
                            0,
                            n_records,
                            0,
                            NULL );

        if( fDataL2 != NULL )
        {
            printf( "L2 day [0] = %"PRId32"\n", fDataL2[ 0 ].j_Day );
            printf( "L2 day [1] = %"PRId32"\n", fDataL2[ 1 ].j_Day );

            printf( "\n\n ** DUMP[0] ** \n\n" );
            vDump_L2IData( &fDataL2[ 0 ], NULL );
            printf( "\n\n ** DUMP[1] ** \n\n" );
            vDump_L2IData( &fDataL2[ 1 ], NULL );

        }
        else
        {
            printf( "Unable to read from file.\n" );
            return EXIT_FAILURE;
        }

        free( fDataL2 );

        vCSFreeFileHandle( fHandle );
    }
    else
    {
        printf( "Unable to open file.\n" );
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
} /* csarray */

