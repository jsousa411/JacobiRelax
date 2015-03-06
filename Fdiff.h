#ifndef _FDIFF_H_
#define _FDIFF_H_

#define dataAt(DATA, I, J, W) DATA[(I) * (W) + J]

void updateGrid(double *, double *, int, int, int, int);
void printGrid(double *, int);
void printMid(double g[], int w, int r);
void initGrid(double [], double [], int);
void createChunk_send(double *,double *,int ,int ,int ,int ,int );
void updateChunk(double *, double *, int , int ,int ,int ,int );

int my_rank, P;
#endif
