#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <complex>
#include <cuda.h>
#include <cufft.h> 
#include <math.h>
#include <vector>
#include <stdio.h>
#include <iostream>
//#include <cuPrintf.cu>

#define M_PI  3.1415926

using namespace std;
typedef complex<long double> long_double_complex;
long_double_complex Il=long_double_complex(0.0,1.0);
 
const long double Cl_SI_HBAR           = 1; //1.054571596e-34;
//const long double Cl_SI_H=1.054571596e-34*2.0*M_PI;
const long double Cl_SI_MASS_ELECTRON  = 1; //9.10938188e-31;
const long double Cl_SI_CHARGE_ELECTRON=1;//1.602176462e-19;
//const long double Cl_SI_EPSILON0       =8.854187817e-12;
//const long double Cl_SI_KB             = 1.3806503e-23;
 
 

 
#define BLOCK_DIM  16
#define BLOCK_SIZE 256

long double potential( long double x, long double y );
long double psiInit(long double x, long double y, long double a );
__global__ void  GPUSummation_parallel_partial( const double2  *iA, double2* g_odata, int N );
__global__ void  GPUSummation_serial( const double2  *iA, double2* oC, int N, int it );
__global__ void SetEtElement(double* xGrid, double* yGrid, double2* expEGrid,int NX, int NY, double dt,double Et,int flag);
__device__ void printet(double Et);
__global__ void GPUMatrixElementMultEt(double2* iA, double2* iB, double2* oC, int N, double scale);
__global__ void GPUMatrixElementMult(double2* iA, double2* iB, double2* oC, int N, double scale);

int main( void )
{
   int NT,NX,NY,N,in,it;
   double *xGrid,*yGrid,*kxGrid,*kyGrid,*kxGridShift,*kyGridShift;
   long double x0,y0,x1,y1,DX,DY,dx,dy;
   long double dt,h,m,a,meff;
   int err;
   FILE *fd, *fd3,*fde;
 
   cufftHandle plan;

   double2 *dev_psiPosGrid, *dev_psiPosInitGrid, *dev_psiMomGrid, *dev_expTGrid, *dev_expVGrid, *dev_act ,*dev_psiCor,*dev_g_odata; // device
   double2 *psiPosGrid,*psiPosInitGrid,*psiMomGrid,*expTGrid,*expVGrid,*act,*psiCor;
   double2 *dev_expEGrid,*expEGrid;
   double *dev_xGrid,*dev_yGrid;
   size_t sizeN,sizeNT;
   clock_t c0,c1;
 
   NX=1024/2; // grid points in x-direction
   NY=1024/2; // grid points in y-direction
 
   DX=400/2; //0.4E-6; // half-width of potential in x-direction
   DY=400/2; //0.4E-6; // half-width of potential in y-direction
 
   dx=2.0*DX/(double)(NX); // grid step size in x-direction
   dy=2.0*DY/(double)(NY); // grid step size in y-direction
   x0=-DX; // lower left corner x-coordinate
   y0=-DY; // lower left corner y-coordinate
   x1=x0+2.0*DX; // upper right corner x-coordinate
   y1=y0+2.0*DY; // upper right corner y-coordinate
   N=NX*NY; // total number of grid points
 
   NT=1000; // number of time-propagtion steps
   dt=0.1; //100.0E-15; // time step
 
   meff=1.0; //0.067; // effective mass
   a=1.E0; //80.0E-9; // gaussian width of initial wavepacket
   h=Cl_SI_HBAR; // hbar
   m=meff*Cl_SI_MASS_ELECTRON; // electron mass in kg
 

   sizeN  = N  * sizeof(double2);
   sizeNT = NT * sizeof(double2);
   psiPosGrid     = (double2*)malloc(sizeN);
   psiPosInitGrid = (double2*)malloc(sizeN);
   psiMomGrid     = (double2*)malloc(sizeN);
   psiCor         = (double2*)malloc(sizeN);
   expTGrid       = (double2*)malloc(sizeN);
   expVGrid       = (double2*)malloc(sizeN);
   expEGrid       = (double2*)malloc(sizeN); //
   act            = (double2*)malloc(sizeNT);
 
   cudaMalloc((void**)&dev_psiPosGrid,sizeN);
   cudaMalloc((void**)&dev_psiPosInitGrid,sizeN);
   cudaMalloc((void**)&dev_psiMomGrid,sizeN);
   cudaMalloc((void**)&dev_psiCor,sizeN);
   cudaMalloc((void**)&dev_expTGrid,sizeN);
   cudaMalloc((void**)&dev_expVGrid,sizeN);
   cudaMalloc((void**)&dev_expEGrid,sizeN); //

   cudaMalloc((void**)&dev_xGrid,sizeof(double)*NX); //   ------------------------------
   cudaMalloc((void**)&dev_yGrid,sizeof(double)*NY); //


   cudaMalloc((void**)&dev_act,sizeNT);
   cudaMalloc((void**)&dev_g_odata,BLOCK_SIZE);
   cufftPlan2d(&plan, NX, NY, CUFFT_Z2Z);

   // initialize the position space grid
   // initialize the momentum space grid and shift it
   xGrid      = (double*) malloc(sizeof(double)*NX);
   kxGrid     = (double*) malloc(sizeof(double)*NX);
   kxGridShift= (double*) malloc(sizeof(double)*NX);
   kyGrid     = (double*) malloc(sizeof(double)*NY);
   kyGridShift= (double*) malloc(sizeof(double)*NY);
   yGrid      = (double*) malloc(sizeof(double)*NY);
   for(int ix=0;ix<NX;ix++)
   {
      xGrid[ix]=x0+ix*dx;
      kxGrid[ix]=-M_PI/dx+double(ix)*2.0*M_PI/double(NX)/dx;
   }
   for(int ix=0;ix<NX/2;ix++)
   {
      kxGridShift[ix]=kxGrid[NX/2+ix];
   }
   for(int ix=NX/2;ix<NX;ix++)
   {
      kxGridShift[ix]=kxGrid[ix-NX/2];
   }
   for(int iy=0;iy<NY;iy++)
   {
      yGrid[iy]=y0+iy*dy;
      kyGrid[iy]=-M_PI/dy+double(iy)*2.0*M_PI/double(NY)/dy;
   }
   for(int iy=0;iy<NY/2;iy++)
   {
      kyGridShift[iy]=kyGrid[NY/2+iy];
   }
   for(int iy=NY/2;iy<NY;iy++)
   {
      kyGridShift[iy]=kyGrid[iy-NY/2];
   }


   double Norm=0.0;
   for(int iy=0;iy<NY;iy++)
   {
      for(int ix=0;ix<NX;ix++)
      {
         int in=ix*NY+iy;
         // do all intermediate calculations in long double to avoid any out of range errors, which DO happen if one uses double for the exp()
         long double V=potential(xGrid[ix],yGrid[iy]);
         long_double_complex expV=exp(Il*(long double)(-(V)*dt));
         long_double_complex expT=exp(Il*(long double)(-(kxGridShift[ix]*kxGridShift[ix]/(2.0l*m)+kyGridShift[iy]*kyGridShift[iy]/(2.0l*m))*dt));
         
		 long_double_complex psi=psiInit(xGrid[ix],yGrid[iy],a);
         // demote long double results to double

         expVGrid[in].x=expV.real();
         expVGrid[in].y=expV.imag();
         expTGrid[in].x=expT.real();
         expTGrid[in].y=expT.imag();
         psiPosGrid[in].x=(double)psi.real();
         psiPosGrid[in].y=(double)psi.imag();
         
		 psiPosInitGrid[in].x=(double)psi.real();
         psiPosInitGrid[in].y=(double)psi.imag();
		
		 //Norm += psiPosGrid[in].x * psiPosGrid[in].x + psiPosGrid[in].y * psiPosGrid[in].y;
		Norm+=psi.real()*psi.real()+psi.imag()*psi.imag();
      }
   }
	Norm=sqrt(Norm*dx*dy);
	std::cout<<"Norm is "<<Norm<<endl;   
	
	//Normalized
	for(int iy=0;iy<NY;iy++)
	{
		for(int ix=0;ix<NX;ix++)
		{
			psiPosGrid[in].x/=Norm;
			psiPosGrid[in].y/=Norm;

		}
	}

	Norm=0.0;
	for(int iy=0;iy<NY;iy++)
   {
      for(int ix=0;ix<NX;ix++)
      {
		  int in=ix*NY+iy;
		  Norm += psiPosGrid[in].x * psiPosGrid[in].x + psiPosGrid[in].y * psiPosGrid[in].y;
	  }
   }
	Norm=sqrt(Norm*dx*dy);
	cout<<"Norm is "<<Norm<<" dx:"<<dx<<" dy:"<<dy<<endl;
 
   long double E0=0.1;
   long double omega = 0.057;
   long double times,t0=0,phi=0.0;
   times=t0;
   long double Et =E0*sin(omega*times-phi);
   for(int iy=0;iy<NY;iy++)
   {
      for(int ix=0;ix<NX;ix++)
      {
         int in=ix*NY+iy;		 
		 long double Ve= -Et*xGrid[ix];
		 long_double_complex expE=exp(Il*(long double)(-(Ve)*dt));
		 expEGrid[in].x = expE.real();
		 expEGrid[in].y = expE.imag();
	  }
   }


   for(int it=0;it<NT;it++) // initialize act[].x/.y
   {
      act[it].x=0.0;
      act[it].y=0.0;
   }
   cudaMemcpy(dev_psiPosGrid,psiPosGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_psiMomGrid,psiMomGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_psiPosInitGrid,psiPosInitGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_expTGrid,expTGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_expVGrid,expVGrid,sizeN,cudaMemcpyHostToDevice);

   cudaMemcpy(dev_expEGrid,expEGrid,sizeN,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_xGrid,xGrid,sizeof(double)*NX,cudaMemcpyHostToDevice);
   cudaMemcpy(dev_yGrid,yGrid,sizeof(double)*NY,cudaMemcpyHostToDevice);

   cudaMemcpy(dev_act,act,sizeNT,cudaMemcpyHostToDevice);
 
   fd=fopen("result_ini.dat","w");
   for(in=0;in<N;in+=100)
   {
      fprintf(fd,"GPU psiPosInitGrid[%i]=(%e,%e)\n",in,(double)psiPosInitGrid[in].x,(double)psiPosInitGrid[in].y);
   }
   fclose(fd);
 
   fprintf(stderr,"Initializing finished. Starting timer ...\n");
 

   c0=clock();
   fde=fopen("efield.dat","w");
   fd=fopen("result_outd.dat","w");

   for(it=0;it<=NT*10;it++) //NT<->5 ------------------------------PROPAGATION------------------------------
   {
	  times=dt*it;
	  Et=E0*sin(omega*times+phi);//*sin(M_PI*times/100)*sin(M_PI*times/100);
	  fprintf(fde,"%e %e\n",times,Et);

 /*   for(int iy=0;iy<NY;iy++)
   {
      for(int ix=0;ix<NX;ix++)
      {
         int in=ix*NY+iy;		 
		 long double Ve= -Et*xGrid[ix];
		 long_double_complex expE=exp(Il*(long double)(-(Ve)*dt));
		 expEGrid[in].x = expE.real();
		 expEGrid[in].y = expE.imag();
	  }
   }

      err=cudaMemcpy(dev_expEGrid,expEGrid,sizeN,cudaMemcpyHostToDevice);
	  printf("%d %d ",it,err);*/

	  SetEtElement<<<N/256,256>>>(dev_xGrid, dev_yGrid, dev_expEGrid, NX, NY, dt, Et, 0);

	  cudaThreadSynchronize();
      GPUMatrixElementMult<<<N/256,256>>>(dev_expEGrid,dev_psiPosGrid,dev_psiPosGrid,N,1.0);
      cudaThreadSynchronize();

      GPUMatrixElementMult<<<N/256,256>>>(dev_expVGrid,dev_psiPosGrid,dev_psiPosGrid,N,1.0);
      cudaThreadSynchronize();
      cufftExecZ2Z(plan, dev_psiPosGrid, dev_psiMomGrid, CUFFT_INVERSE);
      cudaThreadSynchronize();
      GPUMatrixElementMult<<<N/256,256>>>(dev_expTGrid,dev_psiMomGrid,dev_psiMomGrid,N,1.0/(double)N);
      cudaThreadSynchronize();
      cufftExecZ2Z(plan, dev_psiMomGrid, dev_psiPosGrid, CUFFT_FORWARD);
      cudaThreadSynchronize();

#if 1
      GPUMatrixElementMult<<<N/256,256>>>(dev_psiPosGrid,dev_psiPosInitGrid,dev_psiCor,N,1.0);
      cudaThreadSynchronize();
      GPUSummation_parallel_partial<<<BLOCK_SIZE,BLOCK_SIZE>>>(dev_psiCor,dev_g_odata,(unsigned int)N);
      cudaThreadSynchronize();
      GPUSummation_serial<<<1,1>>>(dev_g_odata,dev_act,BLOCK_SIZE,it);
      cudaThreadSynchronize();
#endif

#if 1
	 if(it%100==0){
		printf("%d %f\n",it,times);

		cudaMemcpy(psiPosGrid,dev_psiPosGrid,sizeN,cudaMemcpyDeviceToHost);
		for(int idy=0;idy<NY;idy+=2)
		{
		for(int idx=0;idx<NX;idx+=2){
			in=idy*NX+idx;
			fprintf(fd,"%f ",log10(psiPosGrid[in].x*psiPosGrid[in].x+psiPosGrid[in].y*psiPosGrid[in].y));
		}	  
		}
	}
#endif


   }
   fclose(fd);
   c1=clock();
   fclose(fde);
 
   fprintf(stderr,"Propagation took %.2f s\n",(double)(c1-c0)/(double)CLOCKS_PER_SEC);
 
   cudaMemcpy(act,dev_act,sizeNT,cudaMemcpyDeviceToHost);
   cudaMemcpy(psiPosGrid,dev_psiPosGrid,sizeN,cudaMemcpyDeviceToHost);
 
   fd=fopen("result_gpu_act_dp.dat","w");
   // write recorded autocorrelation function at each time-step
   for(it=0;it<NT;it++)
   {
      fprintf(fd,"%e %e %e\n",(double)(it*dt),(double)(act[it].x*dx*dy),(double)(act[it].y*dx*dy));
   }
   fclose(fd);

      fd=fopen("result_out.dat","w");
	 fd3=fopen("result_outxy.dat","w");
/*    for(in=0;in<N;in+=1)
	 {
	   int idx=in-NX*((in)/NX);
	   int idy=(in)/NX;
      //fprintf(fd,"%e %e %e %e\n",(double)xGrid[idx],(double)yGrid[idy],(double)psiPosGrid[in].x,(double)psiPosGrid[in].y);
	  //fprintf(fd,"%e %e\n",(double)psiPosGrid[in].x,(double)psiPosGrid[in].y);
	  fprintf(fd,"%e ",psiPosGrid[in].x*psiPosGrid[in].x+psiPosGrid[in].y*psiPosGrid[in].y);
	  
   }*/
      for(int idy=0;idy<NY;idy+=2)
   {
		for(int idx=0;idx<NX;idx+=2){
			in=idy*NX+idx;
			fprintf(fd,"%f ",(psiPosGrid[in].x*psiPosGrid[in].x+psiPosGrid[in].y*psiPosGrid[in].y));
			fprintf(fd3,"%f %f %f\n ",(double)xGrid[idx],(double)yGrid[idy],log10(psiPosGrid[in].x*psiPosGrid[in].x+psiPosGrid[in].y*psiPosGrid[in].y));
		}	  
		//fprintf(fd,"\n");
		//fprintf(fd3,"\n");
   }
   fclose(fd);  fclose(fd3);
   // all memory frees are missing ...
 
   return 0;
}






 
__global__ void GPUMatrixElementMult(double2* iA, double2* iB, double2* oC, int N, double scale)
{
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
   double2 z;
   if (idx<N)
   {
      z.x = iA[idx].x * iB[idx].x - iA[idx].y * iB[idx].y;
      z.y = iA[idx].x * iB[idx].y + iA[idx].y * iB[idx].x;
      oC[idx].x = z.x *scale;
      oC[idx].y = z.y *scale;
   }
}
 
__global__ void GPUMatrixElementMultEt(double2* iA, double2* iB, double2* oC, int N, double scale)
{
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
   double2 z;
   if (idx<N)
   {
      z.x = iA[idx].x * iB[idx].x - iA[idx].y * iB[idx].y;
      z.y = iA[idx].x * iB[idx].y + iA[idx].y * iB[idx].x;
      oC[idx].x = z.x *scale;
      oC[idx].y = z.y *scale;
   }
}

__device__ void printet(double Et)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
printf("Et= %f %d\n",Et,idx);
}

__global__ void SetEtElement(double* xGrid, double* yGrid, double2* expEGrid,int NX, int NY, double dt,double Et,int flag)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   //double2 z;
   int N=NX*NY;
	//printet(Et);
   // printf("Et= %f %d %d %d %d\n",Et,idx,blockIdx.x,NX,NY);
	//if(idx == 0) {printf("%f, DONE!\n",Et);}
   if (idx<N)
   {	//printf("%d,%d,%d\n", idx,NX,NY);
         //int idx=ix*NY+iy;
		 int ix=idx/NY; //flag=0 x-direction; 1 y-direction;2 cicular (not implemented yet)
		 double Ve= -Et*xGrid[ix];
		 //double2 expE;
		 //expE.x=cos( (Ve*dt));   //exp(Il*(-(Ve)*dt));
		 //expE.y=- sin(Ve*dt); 
		 expEGrid[idx].x =   cos(Ve*dt);// expE.x;
		 expEGrid[idx].y = - sin(Ve*dt);//  expE.y;
   }
}

/* GPU Correlation requires to adapt the "reduction" example from the SDK,
since we have to avoid memory synchronization problems when we write the results
from all threads in a single double */
 
/* slow serial kernel */
__global__ void  GPUSummation_serial( const double2  *iA, double2* oC, int N, int it )
{
    if( threadIdx.x == 0 )
    {
        oC[it].x = 0.0;
        oC[it].y = 0.0;
        for(int idx = 0; idx < N; idx++)
        {
            oC[it].x += iA[idx].x;
            oC[it].y += iA[idx].y;
        }
    }
}
 
__global__ void  GPUSummation_parallel_partial( const double2  *iA, double2* g_odata, int N )
{
    __shared__ double2 sdata[BLOCK_SIZE];
 
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE * gridDim.x;
    double2 accum;
    
    accum.x = iA[idx].x;
    accum.y = iA[idx].y;
    idx+=gridSize;
 
    while (idx < N)
    {
        accum.x += iA[idx].x;
        accum.y += iA[idx].y;
        idx += gridSize;
    }
 
    sdata[tid].x=accum.x;
    sdata[tid].y=accum.y;
 
    __syncthreads();
 
    if (BLOCK_SIZE >= 512) { if (tid < 256) 
    { sdata[tid].x += sdata[tid + 256].x; sdata[tid].y += sdata[tid + 256].y; } __syncthreads(); }
    if (BLOCK_SIZE >= 256) { if (tid < 128) 
    { sdata[tid].x += sdata[tid + 128].x; sdata[tid].y += sdata[tid + 128].y; } __syncthreads(); }
    if (BLOCK_SIZE >= 128) { if (tid <  64) 
    { sdata[tid].x += sdata[tid +  64].x; sdata[tid].y += sdata[tid +  64].y; } __syncthreads(); }
 
    if (tid < 32)
    {
        if (BLOCK_SIZE >=  64) { sdata[tid].x += sdata[tid + 32].x; sdata[tid].y += sdata[tid + 32].y;  __syncthreads(); }
        if (BLOCK_SIZE >=  32) { sdata[tid].x += sdata[tid + 16].x; sdata[tid].y += sdata[tid + 16].y;  __syncthreads(); }
        if (BLOCK_SIZE >=  16) { sdata[tid].x += sdata[tid +  8].x; sdata[tid].y += sdata[tid +  8].y;  __syncthreads(); }
        if (BLOCK_SIZE >=   8) { sdata[tid].x += sdata[tid +  4].x; sdata[tid].y += sdata[tid +  4].y;  __syncthreads(); }
        if (BLOCK_SIZE >=   4) { sdata[tid].x += sdata[tid +  2].x; sdata[tid].y += sdata[tid +  2].y;  __syncthreads(); }
        if (BLOCK_SIZE >=   2) { sdata[tid].x += sdata[tid +  1].x; sdata[tid].y += sdata[tid +  1].y;  __syncthreads(); }
    }
    // write result for this block to global mem 
    if (tid == 0) 
    {
       g_odata[blockIdx.x].x = sdata[0].x; 
       g_odata[blockIdx.x].y = sdata[0].y; 
    }
}
 

 
long double psiInit(long double x, long double y, long double a )
{
   return exp(-(x*x+y*y)/(2.0*a*a))/(a*sqrt(M_PI));
}
 
long double potential( long double x, long double y )
{
   return -1.0/(x*x+y*y+1);
   //return 10000.0*Cl_SI_CHARGE_ELECTRON*x;
}
 
