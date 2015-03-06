/*
 Joao Sousa
  
 03/02/2014
  
This program performs the Jacobi relaxation/finite difference.

*/
/*************************

   File: Fdiff.c
   Compile: mpicc Fdiff.c Fdutils.c -O3 -o Fd -lm
   Use: ./bsub-v6 -n #process --ppn #proc per node ./Fd [input file]

   Performs 4 nearest neighbor updates on 2-D grid
   Input file format:

   # cycles
   width of grid (including boundary)
   # initial data points
   
   3 integers per data point: i and j indices, data


*************************/

#include <stdio.h>
#include <stdlib.h>
#include "Fdiff.h"
#include <mpi.h>
#include <math.h>


//Directions of neighbor
#define WEST  1
#define NORTH 2
#define EAST  3
#define SOUTH 4
#define WHOLE_CHUNK 5


//Main driver of the program
//It allocates memory for the datagrid
//and drives the update cycles.
//updategrid() is called to update the
//the grid of data
//printgrid is called to print the grid
//other functions are also called for
//assistance with the datagrid
//See Fdutils.c for additional functions
//used for assistance
int main(int argc, char **argv) {

  int width; //width of grid
  int chunk_width;//with of a sub grid i.e. chunck
  int numCycles;//number of cycles to execute
  int i, j, m, n;//iterators
 
  //the arrays to hold data grid, columns, and data place holders for messaging
  double *u0, *u1, *tptr,*l_chunk,*r_chunk,*t_chunk,*b_chunk,*msg,*l_msg,*r_msg,*t_msg,*b_msg,*msg_grid;
  int cycle = 0; //cycle iterator
  int numInit;//initial values in datagrid
 
   //Variables used to capture time to measure execution time
  double comm_start, comm_end, update_start, update_end, cycles_start,cycles_end,average_cycle_time,gather_start,gather_end,total_time,gather_total; 
  
  int P,p;//the number of process for whole grid and per row respectively
  int start_row,start_col;//start point of a process
  FILE *fp;//file pointer: used to read data from file
  double inTemp;//temporary place holder


  //MPI request variables to sync messaging between neighbors
  //_s means send _r means receive
  MPI_Request s_top,s_left,s_right,s_bottom,r_top,r_left,r_right,r_bottom;              

   //read the values about the datagrid
   //size of datagrid, number of cycles and 
   //initial values in the datagrid 
   fp = fopen(argv[1], "r");

   fscanf(fp, "%d", &numCycles);
   fscanf(fp, "%d", &width);
   fscanf(fp, "%d", &numInit);
 
   //reduce the number fo cycles by one because
   //the last cycle does not require neighbor updates
   //so an updategrid() call will be done after the 
   //loop is done 
   numCycles -= 1;
   
   //allocate datagrid memory
   u0 = calloc(width * width, sizeof(double));
   u1 = calloc(width * width, sizeof(double));
  
   //initialize datagrid with 0 
   initGrid(u0, u1, width);
 
   //read datagrid data and set the points
   //that should have data other than 0 
   for (n=0; n<numInit; n++) {
    fscanf(fp, "%d%d%lf", &i, &j, &inTemp);
    dataAt(u1, i, j, width) = inTemp;
   }
  
  //Kick off processes
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
 
  //get the side value i.e. length of grid 
  p=(int)sqrt(P);
  //calculate the width of a sub chunck
  chunk_width=(width-2)/p;

  //allocate gather buffer
  msg=calloc(chunk_width*chunk_width,sizeof(double));

  msg_grid = calloc(chunk_width*chunk_width*(P-1),sizeof(double));
  // allocate send buffers
  l_msg=calloc(chunk_width,sizeof(double));
  r_msg=calloc(chunk_width,sizeof(double));
  t_msg=calloc(chunk_width,sizeof(double));
  b_msg=calloc(chunk_width,sizeof(double));

  // allocate receive buffers
  l_chunk=calloc(chunk_width,sizeof(double));   
  r_chunk=calloc(chunk_width,sizeof(double));
  t_chunk=calloc(chunk_width,sizeof(double));
  b_chunk=calloc(chunk_width,sizeof(double));

  //Determine the start position
  if (my_rank*chunk_width<width-2)
    {
		start_row=1;
		start_col=(my_rank*chunk_width)+1;
	}
  else
	 {
		start_row=(my_rank*chunk_width/(width-2))*chunk_width+1;
		start_col=my_rank*chunk_width%(width-2)+1;
	 }
  
//  printf("Initializing Position : Here in process %d , chunk_width = %d,start_row = %d, start_col = %d",my_rank,chunk_width,start_row,start_col);
 

  //If we have one process only... 
  if (P==1)
  {
    if  (my_rank==0){ 
	 
	  //update_start gives the
	  //amount of time it takes to 
	  //execute updategrid() 
	  update_start=MPI_Wtime();
	  //cycle_start is used to measure the 
	  //start time for the whole operation
	  //it gives the total execution time
	  cycles_start = MPI_Wtime();
	 } 
    
   	 //Peform datagrid updates
     for (cycle=0; cycle<numCycles; cycle++)
	 {
	        updateGrid(u0, u1, width,chunk_width,start_row,start_col);
	 		
			//swap pointers after each data upgrid
	        tptr=u0;
	        u0=u1;
        	u1=tptr;
	 }
     //capture the time after update is done
    if  (my_rank==0) 
	  update_end=MPI_Wtime();

  }
  //If we have more than one process
  else
  {
    if  (my_rank==0){ 
	  update_start=MPI_Wtime();
	  //cycle_start is used to measure the 
	  //start time for the whole operation
	  //it gives the total execution time
	  cycles_start = MPI_Wtime();
	 } 
 	
		//Note numCycles is actuall numCycles -1
		//the last iteration does not require 
		//neighbor messagin, so the updategrid is 
		//done after the loop.  The tim measurement
		//is taken at that updategrid as well 
     for (cycle=0; cycle<numCycles; cycle++){
    
	//Leave the lines below commented as
	//they are used for time measurement(s)
    //when needed
	/*	if  (my_rank==0){ 
	  		update_start=MPI_Wtime();
	 	}
	*/ 
        // Update Process's sub-chunk based on its rank.
        updateGrid(u0, u1, width,chunk_width,start_row,start_col);
    /*	if  (my_rank==0){ 
	  		update_end +=MPI_Wtime()-update_start;
		    comm_start = MPI_Wtime();
	 	} 
     */ 
    
	    if (my_rank==0)   // top left
	    {		
				createChunk_send(r_msg,u0,width,chunk_width,start_row,start_col,EAST);
				createChunk_send(b_msg,u0,width,chunk_width,start_row,start_col,SOUTH);
				 // receive from right,bottom
				MPI_Irecv(r_chunk, chunk_width, MPI_DOUBLE, my_rank+1, WEST, MPI_COMM_WORLD,&r_right);
				MPI_Irecv(b_chunk, chunk_width, MPI_DOUBLE, my_rank+p, NORTH, MPI_COMM_WORLD,&r_bottom);

				//send right,botoom				
				MPI_Isend(r_msg, chunk_width, MPI_DOUBLE, my_rank+1, EAST, MPI_COMM_WORLD, &s_right);
				MPI_Isend(b_msg, chunk_width, MPI_DOUBLE, my_rank+p, SOUTH, MPI_COMM_WORLD, &s_bottom);				
				
  				// wait for all receives
				MPI_Wait(&r_right, NULL);
				MPI_Wait(&r_bottom, NULL);

				 // update right,bottom chunk 
				updateChunk(u0,r_chunk,width,chunk_width,start_row,start_col,EAST);
				updateChunk(u0,b_chunk,width,chunk_width,start_row,start_col,SOUTH);			

				//wait for all sends
				MPI_Wait(&s_right, NULL);
				MPI_Wait(&s_bottom, NULL); 										
		}	
		else if (my_rank==p-1)  //top right
		{	
				createChunk_send(l_msg,u0,width,chunk_width,start_row,start_col,WEST);
				createChunk_send(b_msg,u0,width,chunk_width,start_row,start_col,SOUTH);
				 // receive from left,bottom
				MPI_Irecv(l_chunk, chunk_width, MPI_DOUBLE, my_rank-1, EAST, MPI_COMM_WORLD,&r_left);
				MPI_Irecv(b_chunk, chunk_width, MPI_DOUBLE, my_rank+p, NORTH, MPI_COMM_WORLD,&r_bottom);

				//send left,botoom
				MPI_Isend(l_msg, chunk_width, MPI_DOUBLE, my_rank-1, WEST, MPI_COMM_WORLD, &s_left);
				MPI_Isend(b_msg, chunk_width, MPI_DOUBLE, my_rank+p, SOUTH, MPI_COMM_WORLD, &s_bottom);
					
  				// wait for all receives
				MPI_Wait(&r_left, NULL);
				MPI_Wait(&r_bottom, NULL);

				 // update left,bottom chunk updateChunk();
				updateChunk(u0,l_chunk,width,chunk_width,start_row,start_col,WEST);
				updateChunk(u0,b_chunk,width,chunk_width,start_row,start_col,SOUTH);

				//wait for all sends
				MPI_Wait(&s_left, NULL);
				MPI_Wait(&s_bottom, NULL);							
		}	
		else if (my_rank==P-p)  // bottom left
		{	
				createChunk_send(t_msg,u0,width,chunk_width,start_row,start_col,NORTH);
				createChunk_send(r_msg,u0,width,chunk_width,start_row,start_col,EAST);
				 // receive from top,right
				MPI_Irecv(t_chunk, chunk_width, MPI_DOUBLE, my_rank-p, SOUTH, MPI_COMM_WORLD,&r_top);
				MPI_Irecv(r_chunk, chunk_width, MPI_DOUBLE, my_rank+1, WEST, MPI_COMM_WORLD,&r_right);

				//send top,right
				MPI_Isend(t_msg, chunk_width, MPI_DOUBLE, my_rank-p, NORTH, MPI_COMM_WORLD, &s_top);
				MPI_Isend(r_msg, chunk_width, MPI_DOUBLE, my_rank+1, EAST, MPI_COMM_WORLD, &s_right);
					
  				// wait for all receives
				MPI_Wait(&r_top, NULL);
				MPI_Wait(&r_right, NULL);

				 // update top,right chunk updateChunk();
				updateChunk(u0,t_chunk,width,chunk_width,start_row,start_col,NORTH);
				updateChunk(u0,r_chunk,width,chunk_width,start_row,start_col,EAST);			

				//wait for all sends
				MPI_Wait(&s_top, NULL);
				MPI_Wait(&s_right, NULL);								
		}			
		else if (my_rank==P-1) // bottom right
		{	
				createChunk_send(l_msg,u0,width,chunk_width,start_row,start_col,WEST);
				createChunk_send(t_msg,u0,width,chunk_width,start_row,start_col,NORTH);
				 // receive from left,top
				MPI_Irecv(l_chunk, chunk_width, MPI_DOUBLE, my_rank-1, EAST, MPI_COMM_WORLD,&r_left);
				MPI_Irecv(t_chunk, chunk_width, MPI_DOUBLE, my_rank-p, SOUTH, MPI_COMM_WORLD,&r_top);
				//send left,top
				MPI_Isend(l_msg, chunk_width, MPI_DOUBLE, my_rank-1, WEST, MPI_COMM_WORLD, &s_left);
				MPI_Isend(t_msg, chunk_width, MPI_DOUBLE, my_rank-p, NORTH, MPI_COMM_WORLD, &s_top);
					
  				// wait for all receives
					MPI_Wait(&r_left, NULL);
				        MPI_Wait(&r_top, NULL);

				 // update left,top chunk updateChunk();
				updateChunk(u0,l_chunk,width,chunk_width,start_row,start_col,WEST);
				updateChunk(u0,t_chunk,width,chunk_width,start_row,start_col,NORTH);			

				//wait for all sends
					MPI_Wait(&s_left, NULL);
				        MPI_Wait(&s_top, NULL);
		}			
		else if  (my_rank<p && my_rank!=0)  // first row
		{	
				createChunk_send(l_msg,u0,width,chunk_width,start_row,start_col,WEST);
				createChunk_send(r_msg,u0,width,chunk_width,start_row,start_col,EAST);
				createChunk_send(b_msg,u0,width,chunk_width,start_row,start_col,SOUTH);
				 // receive from left,right,bottom
				MPI_Irecv(l_chunk, chunk_width, MPI_DOUBLE, my_rank-1, EAST, MPI_COMM_WORLD,&r_left);
				MPI_Irecv(r_chunk, chunk_width, MPI_DOUBLE, my_rank+1, WEST, MPI_COMM_WORLD,&r_right);
				MPI_Irecv(b_chunk, chunk_width, MPI_DOUBLE, my_rank+p, NORTH, MPI_COMM_WORLD,&r_bottom);

				//send left,right,botoom
				MPI_Isend(l_msg, chunk_width, MPI_DOUBLE, my_rank-1, WEST, MPI_COMM_WORLD, &s_left);
				MPI_Isend(r_msg, chunk_width, MPI_DOUBLE, my_rank+1, EAST, MPI_COMM_WORLD, &s_right);
				MPI_Isend(b_msg, chunk_width, MPI_DOUBLE, my_rank+p, SOUTH, MPI_COMM_WORLD, &s_bottom);
					
  				// wait for all receives
				MPI_Wait(&r_left, NULL);
				MPI_Wait(&r_right, NULL);
				MPI_Wait(&r_bottom, NULL);

				 // update left,right,bottom chunk updateChunk();
				updateChunk(u0,l_chunk,width,chunk_width,start_row,start_col,WEST); updateChunk(u0,r_chunk,width,chunk_width,start_row,start_col,EAST);
				updateChunk(u0,b_chunk,width,chunk_width,start_row,start_col,SOUTH);

				//wait for all sends				
				MPI_Wait(&s_left, NULL);
				MPI_Wait(&s_right, NULL);
				MPI_Wait(&s_bottom, NULL);
						
		}	
		else if (my_rank%p==0 && my_rank!=0) 	//first column
		{		
				createChunk_send(t_msg,u0,width,chunk_width,start_row,start_col,NORTH);
				createChunk_send(r_msg,u0,width,chunk_width,start_row,start_col,EAST);
				createChunk_send(b_msg,u0,width,chunk_width,start_row,start_col,SOUTH);
				 // receive from top,right,bottom
				MPI_Irecv(t_chunk, chunk_width, MPI_DOUBLE, my_rank-p, SOUTH, MPI_COMM_WORLD,&r_top);
				MPI_Irecv(r_chunk, chunk_width, MPI_DOUBLE, my_rank+1, WEST, MPI_COMM_WORLD,&r_right);
				MPI_Irecv(b_chunk, chunk_width, MPI_DOUBLE, my_rank+p, NORTH, MPI_COMM_WORLD,&r_bottom);

				//send top,right,botoom
				MPI_Isend(t_msg, chunk_width, MPI_DOUBLE, my_rank-p, NORTH, MPI_COMM_WORLD, &s_top);
				MPI_Isend(r_msg, chunk_width, MPI_DOUBLE, my_rank+1, EAST, MPI_COMM_WORLD, &s_right);
				MPI_Isend(b_msg, chunk_width, MPI_DOUBLE, my_rank+p, SOUTH, MPI_COMM_WORLD, &s_bottom);
 					
  				// wait for all receives
				MPI_Wait(&r_top, NULL);
				MPI_Wait(&r_right, NULL);
				MPI_Wait(&r_bottom, NULL);

				 // update top,right,bottom chunk updateChunk();
				updateChunk(u0,t_chunk,width,chunk_width,start_row,start_col,NORTH);
				updateChunk(u0,r_chunk,width,chunk_width,start_row,start_col,EAST);
				updateChunk(u0,b_chunk,width,chunk_width,start_row,start_col,SOUTH);				

				//wait for all sends
				MPI_Wait(&s_top, NULL);
				MPI_Wait(&s_right, NULL);
				MPI_Wait(&s_bottom, NULL);
		}
		else if (my_rank>P-p) // last row
		{		
				createChunk_send(l_msg,u0,width,chunk_width,start_row,start_col,WEST);
				createChunk_send(t_msg,u0,width,chunk_width,start_row,start_col,NORTH);
				createChunk_send(r_msg,u0,width,chunk_width,start_row,start_col,EAST);
				 // receive from left,top,right
				MPI_Irecv(l_chunk, chunk_width, MPI_DOUBLE, my_rank-1, EAST, MPI_COMM_WORLD,&r_left);
				MPI_Irecv(t_chunk, chunk_width, MPI_DOUBLE, my_rank-p, SOUTH, MPI_COMM_WORLD,&r_top);
				MPI_Irecv(r_chunk, chunk_width, MPI_DOUBLE, my_rank+1, WEST, MPI_COMM_WORLD,&r_right);

				//send left,top,right
				MPI_Isend(l_msg, chunk_width, MPI_DOUBLE, my_rank-1, WEST, MPI_COMM_WORLD, &s_left);
				MPI_Isend(t_msg, chunk_width, MPI_DOUBLE, my_rank-p, NORTH, MPI_COMM_WORLD, &s_top);
				MPI_Isend(r_msg, chunk_width, MPI_DOUBLE, my_rank+1, EAST, MPI_COMM_WORLD, &s_right);
 					
  				// wait for all receives
				MPI_Wait(&r_left, NULL);
				MPI_Wait(&r_top, NULL);
				MPI_Wait(&r_right, NULL);

				 // update left,top,right chunk updateChunk();
				updateChunk(u0,l_chunk,width,chunk_width,start_row,start_col,WEST);
				updateChunk(u0,t_chunk,width,chunk_width,start_row,start_col,NORTH);
				updateChunk(u0,r_chunk,width,chunk_width,start_row,start_col,EAST);		

				//wait for all sends
				MPI_Wait(&s_left, NULL);
				MPI_Wait(&s_top, NULL);
				MPI_Wait(&s_right, NULL);			
		}
    	else if (my_rank%p==p-1)  //last column	 
		{	
				createChunk_send(l_msg,u0,width,chunk_width,start_row,start_col,WEST);
				createChunk_send(t_msg,u0,width,chunk_width,start_row,start_col,NORTH);
				createChunk_send(b_msg,u0,width,chunk_width,start_row,start_col,SOUTH);
				 // receive from left,top,bottom
				MPI_Irecv(l_chunk, chunk_width, MPI_DOUBLE, my_rank-1, EAST, MPI_COMM_WORLD,&r_left);
				MPI_Irecv(t_chunk, chunk_width, MPI_DOUBLE, my_rank-p, SOUTH, MPI_COMM_WORLD,&r_top);
				MPI_Irecv(b_chunk, chunk_width, MPI_DOUBLE, my_rank+p, NORTH, MPI_COMM_WORLD,&r_bottom);

				//send left,top,bottom
				MPI_Isend(l_msg, chunk_width, MPI_DOUBLE, my_rank-1, WEST, MPI_COMM_WORLD, &s_left);
				MPI_Isend(t_msg, chunk_width, MPI_DOUBLE, my_rank-p, NORTH, MPI_COMM_WORLD, &s_top);
				MPI_Isend(b_msg, chunk_width, MPI_DOUBLE, my_rank+p, SOUTH, MPI_COMM_WORLD, &s_bottom);
 					
  				// wait for all receives
				MPI_Wait(&r_left, NULL);
				MPI_Wait(&r_top, NULL);
				MPI_Wait(&r_bottom, NULL);

				 // update left,top,bottom chunk updateChunk();
				updateChunk(u0,l_chunk,width,chunk_width,start_row,start_col,WEST);
				updateChunk(u0,t_chunk,width,chunk_width,start_row,start_col,NORTH);
				updateChunk(u0,b_chunk,width,chunk_width,start_row,start_col,SOUTH);

				//wait for all sends
				MPI_Wait(&s_left, NULL);
				MPI_Wait(&s_top, NULL);
				MPI_Wait(&s_bottom, NULL);

    	} 
		else	   // everyother internal element
		{
				createChunk_send(l_msg,u0,width,chunk_width,start_row,start_col,WEST);
				createChunk_send(t_msg,u0,width,chunk_width,start_row,start_col,NORTH);
				createChunk_send(r_msg,u0,width,chunk_width,start_row,start_col,EAST);
				createChunk_send(b_msg,u0,width,chunk_width,start_row,start_col,SOUTH);

				 // receive from left,top,right,bottom
				MPI_Irecv(l_chunk, chunk_width, MPI_DOUBLE, my_rank-1, EAST, MPI_COMM_WORLD,&r_left);
				MPI_Irecv(t_chunk, chunk_width, MPI_DOUBLE, my_rank-p, SOUTH, MPI_COMM_WORLD,&r_top);
				MPI_Irecv(r_chunk, chunk_width, MPI_DOUBLE, my_rank+1, WEST, MPI_COMM_WORLD,&r_right);
				MPI_Irecv(b_chunk, chunk_width, MPI_DOUBLE, my_rank+p, NORTH, MPI_COMM_WORLD,&r_bottom);

				//send left,top,right,botoom
				MPI_Isend(l_msg, chunk_width, MPI_DOUBLE, my_rank-1, WEST, MPI_COMM_WORLD, &s_left);
				MPI_Isend(t_msg, chunk_width, MPI_DOUBLE, my_rank-p, NORTH, MPI_COMM_WORLD, &s_top);
				MPI_Isend(r_msg, chunk_width, MPI_DOUBLE, my_rank+1, EAST, MPI_COMM_WORLD, &s_right);
				MPI_Isend(b_msg, chunk_width, MPI_DOUBLE, my_rank+p, SOUTH, MPI_COMM_WORLD, &s_bottom);
 					
  				// wait for all receives
				MPI_Wait(&r_left, NULL);
				MPI_Wait(&r_top, NULL);
				MPI_Wait(&r_right, NULL);
				MPI_Wait(&r_bottom, NULL);

				 // update left,top,right,bottom chunk updateChunk();
				updateChunk(u0,l_chunk,width,chunk_width,start_row,start_col,WEST);
				updateChunk(u0,t_chunk,width,chunk_width,start_row,start_col,NORTH);
				updateChunk(u0,r_chunk,width,chunk_width,start_row,start_col,EAST);
				updateChunk(u0,b_chunk,width,chunk_width,start_row,start_col,SOUTH);

				//wait for all sends
			 	MPI_Wait(&s_left, NULL);
				MPI_Wait(&s_top, NULL);
				MPI_Wait(&s_right, NULL);
				MPI_Wait(&s_bottom, NULL);

        }	
     // }
      	tptr=u0;
      	u0=u1;
      	u1=tptr;   
      	/*if  (my_rank==0){ 
		    comm_end += MPI_Wtime()-comm_start;
	  	} 
		*/
    }    
 	    /* if  (my_rank==0){ 
		    update_start = MPI_Wtime();
	  		} 
		*/ 
     updateGrid(u0, u1, width,chunk_width,start_row,start_col);
     /*if  (my_rank==0){ 
		update_end -= MPI_Wtime()-update_start;
	 	} 
  	 */ 
	 tptr=u0;
     u0=u1;
     u1=tptr;   
  }
   
  //gather all the chuncks from all other rocess
  if(my_rank == 0)
  {
	//leave code below commented out as it is used
	//for time measurments when needed
	/*
    cycles_end=MPI_Wtime();
    average_cycle_time = ( cycles_end - cycles_start ) / numCycles;
    gather_start=MPI_Wtime();   
	*/


    // Start gathering chunks from P-1 process to Process 0
    MPI_Request gather_chunk[P-1];
    for(n = 1; n < P; n++)
    {
		//receive the chunck of data from each process
        MPI_Irecv(msg_grid+(n-1)*chunk_width*chunk_width, chunk_width*chunk_width, MPI_DOUBLE, n, n, MPI_COMM_WORLD, &gather_chunk[n-1]);
    }

	//Sync up all the process' data received and update each 
	//chunck into the main datagrid: u1
    for (n=1;n<P;n++)
    {
        start_row=(n*chunk_width/(width-2))*chunk_width+1;
        start_col=n*chunk_width%(width-2)+1;
        MPI_Wait(&gather_chunk[n-1], NULL);
        updateChunk(u1,msg_grid+(n-1)*chunk_width*chunk_width,width,chunk_width,start_row,start_col,WHOLE_CHUNK);  
    }
	
	//Peform the last time measurements needed for reporting
    gather_end=MPI_Wtime();
    gather_total=gather_end-gather_start;
    total_time=gather_end-cycles_start;
    update_end = update_end/(numCycles+1);
    comm_end = comm_end / (numCycles+1);
   
	//Leave commented line below because these are used at different runs
   //to measure the time 
	//printf("Dataset :  %s #Cycles : %d #Process : %d  Average Cycle Time : %f \nGather Time : %f Total Time : %f Update: %f Communicate: %f\n",argv[1], numCycles, P, average_cycle_time, gather_total, total_time,update_end, comm_end);
	printf("\nTotal time: %f\n",total_time);
  }
  //if you are not process with rank = 0 then you need send your 
  //chunck of data process 0
  else
  {
    createChunk_send(msg,u1,width,chunk_width,start_row,start_col,WHOLE_CHUNK);
    //send chunk    
    MPI_Send(msg, chunk_width*chunk_width, MPI_DOUBLE,
             0, my_rank, MPI_COMM_WORLD);
   }
  //output the datagrid to the output dump.out file for correctness checking 
  if(my_rank == 0){
    dumpGrid(u1, width);
  }
  
  //That's all folks! 
  MPI_Finalize();
  
}

