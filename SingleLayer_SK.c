/* SINGLE-LAYER SKYRMIONS: MICROMAGNETIC SIMULATION INTEGRATING 
LANDAU-LIFSHITZ-GILBERT EQUATION*/
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
//#include <omp.h> // parallell computing

//*************  Magnetic parameters (SI)  *******************//
/* Here we basically tell the program the amplitude of our interactions */

#define A       15.0e-12  // Exchange interaction amplitude in J/m
#define D1      3.05e-3   // DMI parameter in J/m2 for the layer (typical value: 3.05e-3)
#define Ka      0.6e6     // Anisotropy parameter in J/m3
#define Ms      0.58e6    // Saturation magnetization in A/m
#define Bext    0.0       // External magnetic field (in Tesla)
#define alpha   0.3       // Gilbert damping factor (dimensionless) (typical values we use: 0.001, 0.02, 0.3...)
//#define MAX_THREADS 2   // Set the maximum number of threads (use for parallell computing)

// Math and physical constants
#define M_PI    3.14159265358979323846
#define gamma   1.7595e11      // Gyromagnetic ratio in rad/T.s
#define mu0     12.56637e-7    // Vacuum permeability in J/m.A
#define qe      1.60217663e-19 // charge of electron in C

// Lattice parameters
/* Here we divide the space in cubic cells of sizes (ax,ay,az). Typically our dinamics
occurs only in 2D (since we are considering ultrathin films). Because of that, the 
behavior in the z-direction is basically uniform*/

#define ax      1e-9        // cell size in m
#define ay      1e-9        // cell size in m
#define az      0.4e-9      // FM layer thickness in m

// Magnetization matrix size
#define Nx      200         // Lateral number of grid points in the lattice
#define Ny      200         // Lateral number of grid points in the lattice

/* Time step (typical: 0.001). 
Maybe you need a smaller value to see details for small alpha (e.g., 0.001)*/
#define dt      0.001       

/* Maximum number of time steps 
(a huge limit to end the simulation if our criteria described below is not fullfiled)*/
#define Nt      20000000    

/* Choose 1 for "true" LLG, or 0, for spin relaxation
If you choose '0', then you discard one term of the LLG equation, which
is the one regarding the magnetic moment precession around the effective B.
You can set '0' when you are interested in equilibrium configurations, because
the system will reach equilibrium much faster. */
#define TrueLLG 0           

// *************************************
// Dimensionless parameters below
// ***************************************
double Ener; 
double Jx; double Jy;
double Dx; double Dy;
double K; double B; double Eoax;
double mxo[Nx][Ny], myo[Nx][Ny], mzo[Nx][Ny];     // old component of m for top layer
double mz_initial[Nx][Ny];
double En1;

// Convergence test - equilibrium configuration

/* Here, we set some parameters to tell the program when we are satisfied with the
magnetic configuration and, so, we can stop the simulation.
The variable 'epsm' is the criteria: if the difference between the magnetic configuration
between 'ntest' time steps is smaller than 'epsm', then the simulation ends.*/

int ntest;
int TimeData = 10000;  // Time interval to export magnetization matrices to .dat files (.dat files are quite similar to .txt)
int TimePrint = 10000; // Time interval to print status in terminal (for you to see how things are going on)
double epsm=1.0e-5;    // convergence error criteria

char label1[128], label2[128], label3[128], label4[128],label5[128],label6[128];

//************ Declaring additional functions we use throughout the program ************

// Function to integrate LLG equation and find stable configuration
void LLG(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny],int nn1, int kk); 
// Function to initiallize the spins in the layer
void Init_spins(int nn, int ns, double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny],
                double nx[Nx][Ny], double ny[Nx][Ny], double nz[Nx][Ny]); // m initial values
// Function to export the magnetization matrices to .dat files
void PrintSpins(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny],
                char fname1[], char fname2[], char fname3[]); // Stores each component of the magnetic moments in a file
// Function to export ONLY the z component of magnetization to .dat files
void Print_mz(double dmz[Nx][Ny], char fname4[]);
// Function to assign Periodic Boundary Conditions to the system (virtually infinite system) in x and y directions
void PBCxy(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny]); 
// Function to calculate the energy
void Energy(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny]);
//************************************************************************//

// Function to read a matrix from a existing file, if you have one
void readMatrixFromFile(const char* filename, double matrix[Nx][Ny]) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }
    // Read the matrix values - including ghost border
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            fscanf(file, "%lf", &matrix[j][i]);
        }
    }
    fclose(file);
}

// Main function
int main()
{
  // Number of time steps to run the convergence test (typically 500, for dt=0.001)
  double ntestaux = (0.001/dt)*500;
  ntest=ntestaux;  
  
  //omp_set_num_threads(MAX_THREADS); // this is a code for paralell computing

  /////////////////////////// CHANGE OF UNITS ///////////////////////////
  /* In order to make the LLG equation quantities dimensionless, we change the units.
  Please, be aware of this when you print your results of energy, mag. field,...
  
  - Our length unit is ----> ax (or ay) <-----
  - Our time unit is ----> 1/(gamma*mu0*Ms) <----- this is due to a small algebra in the original LLG equation
  */
  
  // Energy density unit (E0 - energy per area)
  const double Eo = mu0*Ms*Ms; 
  // Energy unit 
  Eoax=Eo*ax*ax*1000000000.0;  

  // Exchange (intralayer - spins of same layer) interaction
  // We consider a spatially homogenous interaction (Jx=Jy)
  Jx = 2*A / (ax*ax*Eo); Jy = 2*A / (ay*ay*Eo);
   
  // DMI constant (also spatially homogenous)
  Dx = D1 / (ax*Eo); Dy = D1 / (ay*Eo);
  
  // Anisotropy and external magnetic field
  K  = 2*Ka / Eo; 
  B = Bext/Eo;
/////////////////////////////////////////////////////////////////////////////////

  // ********** Some announcements in the beginning of the program **************
  printf("\n\n############################################\n");
  printf("SINGLE-LAYER DYNAMICS \n\n");
  printf("Magnetic parameters (SI):\n");
  printf("Jx=Jy=%.2f 10^-12 J/m | DMI: Dx=Dy=%.2f 10^-3 J/m² | K=%.2f 10^6 J/m³ \nB=%.2f T | Ms = %.4f 10^6 A/m \n", A*1e12, D1*1e3,Ka*1e-6,Bext,Ms*1e-6);
  printf("\nLattice parameters: \n");
  printf("ax=ay=%.2f nm \t Nx=%d \t Ny=%d \t dt=%.7f \n",ax/1e-9,Nx,Ny,dt);
  printf("############################################\n\n\n");
  printf("Simulation started! \n\n");
  // Magnetization matrices 
  double mx[Nx][Ny], my[Nx][Ny], mz[Nx][Ny];   // magnetization matrices components
  double nx[Nx][Ny], ny[Nx][Ny], nz[Nx][Ny];   // auxiliary matrices
  unsigned int i,j,t;
    
  // OBS: output folders are created in the shell script

  /*****************************************************************************
  //  Initial conditions (circular domain or previous config)
      Here you can choose if
      1) you start the simulation with all spins up or down, except for a circular
      domain of opposite spin (in respect to the rest)
      2) Load previous magnetic configuration by importing mx,my,mz files

  //*****************************************************************************/
  // 1) Circular domains
  int kk;             // 1 for top layer and -1 for bottom layer (initial condition)
  int nny=Ny/2-1;     // initial y position for nucleation domain (center of sample)
  int nn1=Nx/2-1;     // initial x position for nucleation domain (location in matrix: [nn1][nny])
  // therefore, center of circular domain is located in matrix at indices: [nn1][nny]
  
  kk=1; // all spins with magnetization +1 (except for flipped circular domain)
  
  // initialization of spins in the layer
  Init_spins(nn1,kk,mx, my, mz, nx, ny, nz);              
  
  // 2) Previous config - remove comment section if you want to load a magnetic config.
  //readMatrixFromFile("./m_initial/mx_initial.dat", mx);
  //readMatrixFromFile("./m_initial/my_initial.dat", my);
  //readMatrixFromFile("./m_initial/mz_initial.dat", mz);
  //*****************************************************************************/

  // Storing initial magnetic moment for convergence test afterwards
  for (i=0;i<Nx;++i) 
  {
    for (j=0;j<Ny;++j) 
    {
          mxo[i][j]=mx[i][j]; myo[i][j]=my[i][j]; mzo[i][j]=mz[i][j];
    }
  }
    
  // Call function to integrate LLG equation
  /* Here, the arguments are the magnetization matrices, the initial position of circular domain
  and the initial state of background spins (kk=1 up or kk=-1 down)*/
  LLG(mx, my, mz,nn1,kk);
  return 0;
}

/*************************************************************/
/*******************  F U N C T I O N S  *********************/
/*************************************************************/
void LLG(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny],int nn1, int kk)
{
  double nx[Nx][Ny], ny[Nx][Ny], nz[Nx][Ny];
  
  // auxiliary ctes and variables
  unsigned int i,j,t; int step;
  const double hdt = 0.5*dt;
  const double C1 = (TrueLLG)? 1.0/(1+alpha*alpha) : 0.0;
  const double C2 = (TrueLLG)? C1*alpha : 1.0;
  const double C1hdt = C1*hdt; const double C2hdt = C2*hdt;
  const double C1dt = C1*dt; const double C2dt = C2*dt;

  // initializing auxiliary matrices
  for (i=0;i<Nx;++i) 
  {
    for (j=0;j<Ny;++j) 
    {
       nx[i][j]=mx[i][j]; ny[i][j]=my[i][j]; nz[i][j]=mz[i][j];
    }
  }


  // Loop in time
  for (t=0;t<Nt;++t)
  {
    /* We have two loops in the matrices (i,j)*/
    //#pragma omp parallel for // this is a command for paralell computing
    for (i=1;i<Nx-1;++i)
    {
      for (j=1;j<Ny-1;++j)
      {
        double sx = mx[i][j]; double sy = my[i][j]; double sz = mz[i][j];
        double two_sx = 2*sx; double two_sy = 2*sy; double two_sz = 2*sz;

        // calculating the effective fields regarding exchange and DMI interaction
        double BJx = Jx*(mx[i-1][j] + mx[i+1][j] - two_sx) + Jy*(mx[i][j-1] + mx[i][j+1] - two_sx);
        double BJy = Jx*(my[i-1][j] + my[i+1][j] - two_sy) + Jy*(my[i][j-1] + my[i][j+1] - two_sy);
        double BJz = Jx*(mz[i-1][j] + mz[i+1][j] - two_sz) + Jy*(mz[i][j-1] + mz[i][j+1] - two_sz);
        double BDx = Dx*(mz[i+1][j] - mz[i-1][j]);
        double BDy = Dy*(mz[i][j+1] - mz[i][j-1]);
        double BDz = Dx*(mx[i-1][j] - mx[i+1][j]) + Dy*(my[i][j-1] - my[i][j+1]);

        // calculating total effective field components
        // the anisotropy only appears in z axis
        double Bx = BJx + BDx + B;
        double By = BJy + BDy + B;
        double Bz = BJz + BDz + B + K*sz;
        
        // calculating auxiliary vector products
        double taux = sy*Bz - sz*By;
        double tauy = sz*Bx - sx*Bz;
        double tauz = sx*By - sy*Bx;
     
        // updating auxiliary matrices
        nx[i][j] = sx - (C1hdt*taux + C2hdt*(sy*tauz - sz*tauy));
        ny[i][j] = sy - (C1hdt*tauy + C2hdt*(sz*taux - sx*tauz));
        nz[i][j] = sz - (C1hdt*tauz + C2hdt*(sx*tauy - sy*taux));

        //  Each time we normalize the magnetizations
        double mag_inv = 1.0/sqrt(nx[i][j]*nx[i][j] + ny[i][j]*ny[i][j] + nz[i][j]*nz[i][j]);
        nx[i][j] *= mag_inv; ny[i][j] *= mag_inv; nz[i][j] *= mag_inv;
      }
    }

    //PBCxy(nx,ny,nz); // if you want PBC, you can use this line

    //#pragma omp parallel for      // this is a command for paralell computing
    // this is the second loop, in which we effectively update magnetization matrices
    for (i=1;i<Nx-1;++i) 
    {
      for (j=1;j<Ny-1;++j) 
      {
          // top layer
          double sx = nx[i][j]; double sy = ny[i][j]; double sz = nz[i][j];
          double two_sx = 2*sx; double two_sy = 2*sy; double two_sz = 2*sz;
          double BJx = Jx*(nx[i-1][j] + nx[i+1][j] - two_sx) + Jy*(nx[i][j-1] + nx[i][j+1] - two_sx);
          double BJy = Jx*(ny[i-1][j] + ny[i+1][j] - two_sy) + Jy*(ny[i][j-1] + ny[i][j+1] - two_sy);
          double BJz = Jx*(nz[i-1][j] + nz[i+1][j] - two_sz) + Jy*(nz[i][j-1] + nz[i][j+1] - two_sz);
          double BDx = Dx*(nz[i+1][j] - nz[i-1][j]);
          double BDy = Dy*(nz[i][j+1] - nz[i][j-1]);
          double BDz = Dx*(nx[i-1][j] - nx[i+1][j]) + Dy*(ny[i][j-1] - ny[i][j+1]);

          double Bx = BJx + BDx + B;
          double By = BJy + BDy + B;
          double Bz = BJz + BDz + B + K*sz;
          double taux = sy*Bz - sz*By;
          double tauy = sz*Bx - sx*Bz;
          double tauz = sx*By - sy*Bx;
          
          mx[i][j] -= (C1dt*taux + C2dt*(sy*tauz - sz*tauy)); 
          my[i][j] -= (C1dt*tauy + C2dt*(sz*taux - sx*tauz));
          mz[i][j] -= (C1dt*tauz + C2dt*(sx*tauy - sy*taux));

          //  Normalization
          double mag_inv = 1.0/sqrt(mx[i][j]*mx[i][j] + my[i][j]*my[i][j] + mz[i][j]*mz[i][j]);
          mx[i][j] *= mag_inv; my[i][j] *= mag_inv; mz[i][j] *= mag_inv;
      }
    }  

    //PBCxy(mx,my,mz); // if you want PBC, you can use this line

    // At each TimeData steps, we print magnetization in the files
    if(t%TimeData == 0)
    {
      char fname1[128]; char fname2[128]; char fname3[128];

      // we create the name of the files for mx,my,mz at time t
      sprintf(fname1,"./mx/mx_t%d.dat", t); 
      sprintf(fname2,"./my/my_t%d.dat", t);
      sprintf(fname3,"./mz/mz_t%d.dat", t); 
    
      // Calling the function which exports the magnetization to files
      PrintSpins(mx,my,mz,fname1,fname2,fname3); 
    }

    // Convergence test - is the simulation over?
    double maxm=0.0;
    if (!(t%ntest) && t>5000)
    { 
      for (i=2;i<Nx-2;++i)
      {
        // Evaluating the difference in magnetization over time - to check if the equilibrium is reached
        for (j=2;j<Ny-2;++j)
        {
            double oldm=fabs(mx[i][j]-mxo[i][j])+fabs(my[i][j]-myo[i][j])+fabs(mz[i][j]-mzo[i][j]);
            if(oldm>maxm)   maxm=oldm;
            mxo[i][j]=mx[i][j]; myo[i][j]=my[i][j]; mzo[i][j]=mz[i][j];                     
        }
      }
      step=t;
      // If the criteria for equilibrium is attended (maxm < epsm), exports the last configuration and end simulation
      if(maxm<epsm)
      {
        t=Nt;
        char ffname1[128],ffname2[128],ffname3[128];

        // we put the final configuration in a separate folder
        sprintf(ffname1,"./m_final/mx_final.dat");
        sprintf(ffname2,"./m_final/my_final.dat");
        sprintf(ffname3,"./m_final/mz_final.dat");
        
        PrintSpins(mx,my,mz,ffname1,ffname2,ffname3); 
      }

    }

    if(t%TimePrint==0) 
    {
        // following simulation at the terminal
        printf("step=%d \t deviate=%.12lf \n",t,maxm);

        // calculating energy for each TimePrint steps
        Energy(mx,my,mz); En1 = Ener;     // calculating layer energy
        printf("Energy = %.4f (in units of Eoax)\n\n",En1);
    }    
  }
}

// Calculating energy for a given ferromagnetic layer
void Energy(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny]) 
{
  int i,j;
  double contEnx=0.0; double contEna=0.0; double contEnD=0.0;
    for (i=1;i<Nx-1;++i) 
    {
        for (j=1;j<Ny-1;++j)
        {
            double sx = mx[i][j]; double sy = my[i][j]; double sz = mz[i][j];
            double BJx = Jx*(mx[i-1][j] + mx[i+1][j] - 2*sx) + Jy*(mx[i][j-1] + mx[i][j+1] - 2*sx);
            double BJy = Jx*(my[i-1][j] + my[i+1][j] - 2*sy) + Jy*(my[i][j-1] + my[i][j+1] - 2*sy);
            double BJz = Jx*(mz[i-1][j] + mz[i+1][j] - 2*sz) + Jy*(mz[i][j-1] + mz[i][j+1] - 2*sz);
            double BDx = Dx*(mz[i+1][j] - mz[i-1][j]);
            double BDy = Dy*(mz[i][j+1] - mz[i][j-1]);
            double BDz = Dx*(mx[i-1][j] - mx[i+1][j]) + Dy*(my[i][j-1] - my[i][j+1]);

            contEnx=contEnx-BJx*mx[i][j]-BJy*my[i][j]-BJz*mz[i][j];
            contEna=contEna+K*(1-mz[i][j]*mz[i][j]);
            contEnD=contEnD-BDx*mx[i][j]-BDy*my[i][j]-BDz*mz[i][j];
        }
    }
    Ener=0.5*(contEnx+contEnD+contEna); // we divide by two to avoid double calculation
}

// Periodic boundary conditions (PBC)
void PBCxy(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny]) 
{
  for (int i=1;i<Ny-1;++i) 
  {
        mx[0][i]    = mx[Nx-2][i];
        mx[Nx-1][i] = mx[1][i];
        my[0][i]    = my[Nx-2][i];
        my[Nx-1][i] = my[1][i];
        mz[0][i]    = mz[Nx-2][i];
        mz[Nx-1][i] = mz[1][i];
  }
  for (int i=1;i<Nx-1;++i) 
  {
        mx[i][0]    = mx[i][Ny-2];
        mx[i][Ny-1] = mx[i][1];
        my[i][0]    = my[i][Ny-2];
        my[i][Ny-1] = my[i][1];
        mz[i][0]    = mz[i][Ny-2];
        mz[i][Ny-1] = mz[i][1];
  }
  // left-bottom corner
  mx[0][0] = mx[Nx-2][Ny-2]; my[0][0] = my[Nx-2][Ny-2]; mz[0][0] = mz[Nx-2][Ny-2];
  // left-top corner
  mx[0][Ny-1] = mx[Nx-2][1]; my[0][Ny-1] = my[Nx-2][1]; mz[0][Ny-1] = mz[Nx-2][1];
  // right-bottom corner
  mx[Nx-1][0] = mx[1][Ny-2]; my[Nx-1][0] = my[1][Ny-2]; mz[Nx-1][0] = mz[1][Ny-2];
  // right-top corner
  mx[Nx-1][Ny-1] = mx[1][1]; my[Nx-1][Ny-1] = my[1][1]; mz[Nx-1][Ny-1] = mz[1][1];
}

// Initializing spins for a given ferromagnetic layer
void Init_spins(int nn, int ns, double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny],
                double nx[Nx][Ny], double ny[Nx][Ny], double nz[Nx][Ny]) {
    unsigned int i,j,ii;
    int nny=Ny/2-1; 
    if(nn<5){
        printf("O nn deve ser maior que 5");
        exit(1);
    }  

    // filling matrix, except ghost border
    for (i=0;i<Nx;++i) {
        for (j=0;j<Ny;++j){
            ii=(i-nn)*(i-nn)+(j-nny)*(j-nny);
            nx[i][j] = mx[i][j] = 0.0;
            ny[i][j] = my[i][j] = 0.0;
            nz[i][j] = mz[i][j] = ns*1.0;
            if(ii<400) nz[i][j] = mz[i][j] = -ns*1.0; // we flip a given region below r^2=400
        }
    }

    // using Open Boundary Conditions
    // creating the "ghost border"
    /* for Open Boundary Conditions, you get the borders of the simulation box
    and assign it zero spin and remove the border in the LLG function*/
    for (i=0;i<Nx;++i) 
    {
        nx[i][0] = mx[i][0] = 0.0;
        ny[i][0] = my[i][0] = 0.0;
        nz[i][0] = mz[i][0] = 0.0;

        nx[i][Ny-1] = mx[i][Ny-1] = 0.0;
        ny[i][Ny-1] = my[i][Ny-1] = 0.0;
        nz[i][Ny-1] = mz[i][Ny-1] = 0.0;
    }

    for (j=0;j<Ny;++j) 
    {
        nx[0][j] = mx[0][j] = 0.0;
        ny[0][j] = my[0][j] = 0.0;
        nz[0][j] = mz[0][j] = 0.0;

        nx[Nx-1][j] = mx[Nx-1][j] = 0.0;
        ny[Nx-1][j] = my[Nx-1][j] = 0.0;
        nz[Nx-1][j] = mz[Nx-1][j] = 0.0;
    }
}

// Exporting mx,my and mz matrices to a .dat file to respective folders
void PrintSpins(double mx[Nx][Ny], double my[Nx][Ny], double mz[Nx][Ny],
                char fname1[], char fname2[], char fname3[]) {
    
    // Opening empty .dat files 
    FILE *fMx = fopen(fname1,"w+");
    if ( fMx == NULL) { 
        fprintf(stderr,"Couldn't open %s\n: %s", fname1, strerror(errno));
        exit(1);
    }
    FILE *fMy = fopen(fname2,"w+");
    if ( fMy == NULL) { 
        fprintf(stderr,"Couldn't open %s\n: %s", fname2, strerror(errno));
        exit(1);
    }
    FILE *fMz = fopen(fname3,"w+");
    if ( fMz == NULL) { 
        fprintf(stderr,"Couldn't open %s\n: %s", fname3, strerror(errno));
        exit(1);
    }

    // we are printing the ghost border, so the matrix
    // actually has a shape (Nx-1,Ny-1)
    for (int j=0;j<Ny;++j)
    {
       for (int i=0;i<Nx;++i)
       {
          fprintf(fMx,"%.12lf\t", mx[i][j]);
          fprintf(fMy,"%.12lf\t", my[i][j]);
          fprintf(fMz,"%.12lf\t", mz[i][j]);
       }
      fprintf(fMx,"\n");
      fprintf(fMy,"\n");
      fprintf(fMz,"\n");
    }
    
    fclose(fMx); fclose(fMy); fclose(fMz);
}

// Printing the mz component alone for a given ferromagnetic layer
// This is an option if you just want to see the z component of magnetization
void Print_mz(double mz[Nx][Ny], char fname4[])
{
    // Opening empty .dat file 
    FILE *fmz = fopen(fname4,"w+");
    if ( fmz == NULL) { 
        fprintf(stderr,"Couldn't open %s\n: %s", fname4, strerror(errno));
        exit(1);
    }
    
    /* we are printing the ghost border, so the magnetization matrix
    we are interesting in actually has a shape (Nx-1,Ny-1)*/
    for (int j=0;j<Ny;++j)
    {
       for (int i=0;i<Nx;++i)
       {
          fprintf(fmz,"%.12lf\t", mz[i][j]);
       }
      fprintf(fmz,"\n");
    }
    fclose(fmz);
}