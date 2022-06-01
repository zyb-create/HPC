static char help[] = "Slove the 1D heat trans explicitly";

#include <petscmat.h>
#include <petscmath.h>
#include <math.h>

int main(int argc,char **args)
{
  Vec            x, u, ut, f, u0;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  PetscReal      norm=0.0,tol=10.*PETSC_MACHINE_EPSILON,normt=1.0;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 101,col[3],its,rstart,rend,nlocal,rank;
  PetscScalar    left = 0.0, right = 1.0, l=2.0, cfl = 0.001;
  PetscScalar    zero = 0.0,value[3],omega = 1.6,dx = (right - left) * 1.0 / (n - 1), kappa = 1.0, dt = dx * cfl, rho=1, c=1, h=0, t=0;
  PetscScalar    pi = M_PI;
  PetscInt       step = 0;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  dx   = (right - left)*1.0 / (n-1);
  dt   = dx * cfl;
  PetscScalar    at = kappa*dt/(rho*c*dx*dx), bt = dt/(rho*c), ct = 2*dx*h/kappa;
    at *= omega; bt *= omega;  // SOR method

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ut);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u0);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  if (!rstart) 
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1 - 2*at; value[1] = 2*at;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(x,i,left,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(f,i,bt*sin(l*pi*(left + i*dx)) + 2*at*ct,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n) 
  {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = 2*at; value[1] = 1 - 2*at;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(x,i,right,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(f,i,bt*sin(l*pi*(left + i*dx)) - 2*at*ct,INSERT_VALUES);CHKERRQ(ierr);
  }

  value[0] = at; value[1] = 1- 2*at; value[2] = at;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(x,i,left + i*dx,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(f,i,bt*sin(l*pi*(left + i*dx)),INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix and vector */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecCopy(x,u);CHKERRQ(ierr);
  ierr = VecExp(u);CHKERRQ(ierr);

  // ierr = VecSet(u,zero);CHKERRQ(ierr);
  while(PetscAbsReal(norm-normt) > tol && step < 1e10){
     ierr = VecCopy(u,ut);CHKERRQ(ierr);
     step = step + 1;
     normt= norm;
     ierr = MatMultAdd(A,ut,f,u);
     ierr = VecCopy(u,u0);CHKERRQ(ierr);
     ierr = VecAXPY(u0,-1,ut);CHKERRQ(ierr);
     ierr = VecNorm(u0,NORM_2,&norm);CHKERRQ(ierr);
    //  ierr = PetscPrintf(PETSC_COMM_WORLD,"step = %d, norm = %g\n",step,(double)norm);CHKERRQ(ierr);
    //  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"step = %d, norm = %g\n",step,(double)norm);CHKERRQ(ierr);
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&ut);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr); ierr = VecDestroy(&u0);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

// EOF
