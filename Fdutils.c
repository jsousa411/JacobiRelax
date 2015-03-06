/*
  Joao Sousa
  
  03/02/2014
 
  This file contains various helper functions to
  assis Fdiff.c with gridupdate, gridprint and other
  functionalities

  compile this file in conjunction with Fdiff.c

  i.e. mpicc Fdiff.c Fdutils.c -o outputfileName 
*/
#include <stdio.h>
#include "Fdiff.h"

/****

     tu[] is input grid
     u[] is output grid
     w is width of grid

     compute 4-nearest neighbor updates 

****/

//Updates the grid based on paramater's values
void updateGrid(double u[], double tu[], int width, int chunk_width,int start_row, int start_col) {
  int i, j;
  int end_row_chunk,end_col_chunk;
    
  end_row_chunk=start_row+chunk_width;				
  end_col_chunk=start_col+chunk_width;
				
  for (i=start_row; i<end_row_chunk; i++) {
    for (j=start_col; j<end_col_chunk; j++) {      
      dataAt(u, i, j, width) = .25 * (dataAt(tu, i+1, j, width)
				  + dataAt(tu, i-1, j, width)
				  + dataAt(tu, i, j+1, width)
				  + dataAt(tu, i, j-1, width));
    }
  }
}

//update's data chunck based on parameters
//The parameter's are sent/received to/from neighbor process
void updateChunk(double u[], double chunk[], int width, int chunk_width,int start_row,int start_col,int direction)
{
    int i,indexR,indexC,row,col,temp_width,flag_width;
    temp_width=chunk_width;
    flag_width=1;

    //direction tells which datapoint can
    // get updated with the data at hand
	//i.e. top, bottom, left or right neighbor
    switch (direction)
    {
	case 1 :  row=1;
                  col=0;
 		  start_col--;
		  break;
	case 2 :  row=0;
                  col=1;
 		  start_row--;		
		  break;
	case 3 :  row=1;
		  col=0;
		  start_col=start_col+chunk_width;
		  break;
	case 4 :  row=0;
		  col=1;
		  start_row=start_row+chunk_width;
		  break;
 
	//this is the condition to copy/update the whole chunck
	//this option is used at the gather step
	//where process 0 gathers every other's process data
	//and syncs it
	case 5 :  temp_width=chunk_width*chunk_width;
		  row=1;col=1;
		  flag_width=chunk_width;
		  break;

	default :break;
   }

	//make the update basd on the settings
	//occured in the switch statement above
   for (i=0;i< temp_width;i++)
   {
		//determine the row index
	    indexR =  i*row/flag_width + start_row;
	    //determine the column index
		indexC = (i*col)%chunk_width + start_col;

		//set the data
		dataAt(u,indexR,indexC,width)=chunk[i];
   }     
}



//Sets the data into a temporary memory, so it can be passed to a neighbor
//if data is in a halo then do nothing
void createChunk_send(double msg[],double u[],int width,int chunk_width,int start_row,int start_col, int direction)
{
	int i,row,col,j;

	//direction indicates which neighbor
	//will receive the data
	//adjacent to current node i.e.
	//top, bottom, left or right node
    switch (direction)
    {
	case 1 :  row=1;
                  col=0;
		  break;
	case 2 :  row=0;
                  col=1;
		  break;
	case 3 :  row=1;
		  col=0;
		  start_col=start_col+chunk_width-1;
		  break;
	case 4 :  row=0;
		  col=1;
		  start_row=start_row+chunk_width-1;
		  break;
	
	//this is the case where the data is not at a corner
	//so we pass in the whole chunck
	//this is used at the gather step
	//where process rank = 0 gathers all the chuncks
	case 5 :  for (i=0;i<chunk_width;i++)
				for (j=0;j<chunk_width;j++)
					msg[i*chunk_width+j]=dataAt(u,i+start_row,j+start_col,width);
		  break;

	default : break;
    }
    
	//pass in a row or a column to the temp array 
	if (direction!=5)
	    for (i=0;i<chunk_width;i++)
			msg[i]=dataAt(u,start_row+row*i,start_col+col*i,width);

}

//prints the whole datagrid to the 
//result file
void printGrid(double g[], int w) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      printf("%7.3f ", dataAt(g, i, j, w));
    }
    printf("\n");
  }
  printf("\n");
}

//Prints a message to the result 
//file
void printMsg(double msg[],int w)
{
   int i,j;
   for (i=0;i<w;i++)
   {
		printf("%7.3f ", msg[i]);
   }
    printf("\n");		
}

//Prints the output datagrid
//to a dump.out file for
//correctness checking
void dumpGrid(double g[], int w) {
  int i, j;
  FILE *fp;

  fp = fopen("dump.out", "w");
  
  for (i=0; i<w; i++) {
    for (j=0; j<w; j++) {
      fprintf(fp, "%f ", dataAt(g, i, j, w));
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}


//Initializes the whole grid to 0
void initGrid(double u0[], double u1[], int w ) {
  int i, j;

  for (i=0; i<w; i++) {
    for (j=0 ; j<w; j++) {
      dataAt(u0, i, j, w) = 0.;
      dataAt(u1, i, j, w) = 0.;
    }
  }
}

