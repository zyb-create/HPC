static char help[] = "Solves a tridiagonal linear system.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

  Note:  The corresponding uniprocessor example is ex1.c
*/
#include <petscksp.h>
#include <petscmath.h>
#include <math.h>

int main(int argc,char **args)
{
  Vec            x, b, u, ut, u0, f;          /* approx solution, RHS, exact solution */
  Mat            A;                /* linear system matrix */
  KSP            ksp;              /* linear solver context */
  PC             pc;               /* preconditioner context */
  PetscReal      norm=0.0,tol=10.*PETSC_MACHINE_EPSILON,normt=1.0;  /* norm of solution error */
  PetscErrorCode ierr;
  PetscInt       i,n = 101,col[3],its,rstart,rend,nlocal,rank;
  PetscScalar    zero = 0.0,value[3],omega = 1.6,dx = 0.01, kappa = 1.0, dt=0.005, rho=1, c=1, h=0, t=0;
  PetscScalar    at = kappa*dt/(rho*c*dx*dx), bt = dt/(rho*c), ct = 2*dx*h/kappa, left = 0, right = 1, l=2.0;
  PetscScalar    pi = M_PI;
  PetscInt       step = 0;

  at *= omega; bt *= omega;  // SOR method

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  dx   = (right - left)*1.0 / (n-1);
  dt   = dx * 0.8;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed. For this simple case let PETSc decide how
     many elements of the vector are stored on each processor. The second
     argument to VecSetSizes() below causes PETSc to decide.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ut);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u0);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&f);CHKERRQ(ierr);

  /* Identify the starting and ending mesh points on each
     processor for the interior part of the mesh. We let PETSc decide
     above. 
   */
  ierr = VecGetOwnershipRange(x,&rstart,&rend);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&nlocal);CHKERRQ(ierr);

  /*
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.

     We pass in nlocal as the "local" size of the matrix to force it
     to have the same parallel layout as the vector created above.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,nlocal,nlocal,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /*
     Assemble matrix.

     The linear system is distributed across the processors by
     chunks of contiguous rows, which correspond to contiguous
     sections of the mesh on which the problem is discretized.
     For matrix assembly, each processor contributes entries for
     the part that it owns locally.
  */
  if (!rstart) 
  {
    rstart = 1;
    i      = 0; col[0] = 0; col[1] = 1; value[0] = 1+2*at; value[1] = -2*at;
    ierr   = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(x,i,left,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(f,i,zero - at*ct,INSERT_VALUES);CHKERRQ(ierr);
  }
  
  if (rend == n) 
  {
    rend = n-1;
    i    = n-1; col[0] = n-2; col[1] = n-1; value[0] = -2*at; value[1] = 1+2*at;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(x,i,right,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValue(f,i,zero + at*ct,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Set entries corresponding to the mesh interior */
  value[0] = -at; value[1] = 1+2*at; value[2] = -at;
  for (i=rstart; i<rend; i++) 
  {
    col[0] = i-1; col[1] = i; col[2] = i+1;
    ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(x,i,left + i*dx,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecSetValue(f,i,bt*sin(l*pi*(left + i*dx)),INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecAssemblyBegin(f);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(f);CHKERRQ(ierr);
  
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecCopy(x,b);CHKERRQ(ierr);

  ierr = VecExp(b);CHKERRQ(ierr);
  // if(rank == 0){
  //    i    = 0;
  //    ierr = VecSetValue(b,i,0,INSERT_VALUES);CHKERRQ(ierr);
  //    i    = n-1;
  //    ierr = VecSetValue(b,i,0,INSERT_VALUES);CHKERRQ(ierr);
  // }  
  // ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  // ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  ierr = VecAXPY(b,1,f);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);

  /*
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       KSPSetFromOptions();
  */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCJACOBI);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  ierr = VecSet(u,zero);CHKERRQ(ierr);
  while(PetscAbsReal(norm-normt) > tol && step < 10000){
     ierr = VecCopy(u,ut);CHKERRQ(ierr);
     step = step + 1;
     normt= norm;
     ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);
     ierr = VecCopy(u,u0);CHKERRQ(ierr);
     ierr = VecAXPY(u0,-1,ut);CHKERRQ(ierr);
     ierr = VecNorm(u0,NORM_2,&norm);CHKERRQ(ierr);
     ierr = VecCopy(u,b);CHKERRQ(ierr);    
    //  if(rank == 0){
    //     i    = 0;
    //     ierr = VecSetValue(b,i,-at*ct,ADD_VALUES);CHKERRQ(ierr);
    //     // ierr = VecSetValue(b,i,-at*ct,INSERT_VALUES);CHKERRQ(ierr);
    //     i    = n-1;
    //     ierr = VecSetValue(b,i,at*ct,ADD_VALUES);CHKERRQ(ierr);
    //     // ierr = VecSetValue(b,i,at*ct,INSERT_VALUES);CHKERRQ(ierr);
    //   }
    //   ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    //   ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
      ierr = VecAXPY(b,1,f);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"step = %d, norm = %g\n",step,(double)norm);CHKERRQ(ierr);
   }
  ierr = VecView(u,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
//   ierr = PetscPrintf(PETSC_COMM_WORLD,"solution = %f\n",1.0/norm);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Solve linear system
  */
//   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Check the error
  */
//   ierr = VecAXPY(x,-1.0,u);CHKERRQ(ierr);
//   ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
//   ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
//   if (norm < tol) {
//     ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);CHKERRQ(ierr);
// //   }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&ut);CHKERRQ(ierr); ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr); ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u0);CHKERRQ(ierr); ierr = VecDestroy(&f);CHKERRQ(ierr);

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
