/*********************************************************
 * sudoku_omp.c
 *
 * A parallel Sudoku solver using backtracking with 
 * OpenMP tasks. Short, human-friendly comments included.
 *********************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <omp.h>
 
 // Each Sudoku board is 9x9.
 #define N 9
 
 // Global flag indicating if we've found a solution.
 static int solutionFound = 0;
 
 // Stores the first valid solution we find.
 static int solvedBoard[N][N];
 
 /*
  * copyBoard:
  *   Copies a 9x9 Sudoku board from 'src' to 'dest',
  *   ensuring parallel tasks don't overwrite each other's data.
  */
 void copyBoard(int src[N][N], int dest[N][N])
 {
     for(int i = 0; i < N; i++){
         for(int j = 0; j < N; j++){
             dest[i][j] = src[i][j];
         }
     }
 }
 
 /*
  * canPlace:
  *   Checks if placing 'digit' at (row,col) is valid:
  *   - Not in same row or column
  *   - Not in the same 3x3 sub-box
  */
 int canPlace(int board[N][N], int row, int col, int digit)
 {
     for(int i = 0; i < N; i++){
         if(board[row][i] == digit) return 0;
         if(board[i][col] == digit) return 0;
     }
     int startRow = (row / 3) * 3;
     int startCol = (col / 3) * 3;
     for(int r = 0; r < 3; r++){
         for(int c = 0; c < 3; c++){
             if(board[startRow + r][startCol + c] == digit){
                 return 0;
             }
         }
     }
     return 1;
 }
 
 /*
  * findNextEmptyCell:
  *   Finds the next empty cell (marked 0),
  *   returns 1 and sets (row,col) if found,
  *   or 0 if there are no more empty cells.
  */
 int findNextEmptyCell(int board[N][N], int *row, int *col)
 {
     for(int r = 0; r < N; r++){
         for(int c = 0; c < N; c++){
             if(board[r][c] == 0){
                 *row = r;
                 *col = c;
                 return 1;
             }
         }
     }
     return 0;
 }
 
 /*
  * backtrack:
  *   Tries placing digits 1..9 in each empty cell.
  *   Uses OpenMP tasks to explore each valid option.
  */
 void backtrack(int board[N][N])
 {
     // Stop if another thread has found a solution.
     if(solutionFound) return;
 
     int row, col;
     // If there's no empty cell, we've solved the puzzle.
     if(!findNextEmptyCell(board, &row, &col)){
         #pragma omp critical
         {
             if(!solutionFound){
                 copyBoard(board, solvedBoard);
                 solutionFound = 1;
             }
         }
         return;
     }
 
     // Try digits 1 through 9 in this empty cell.
     for(int digit = 1; digit <= 9; digit++){
         if(canPlace(board, row, col, digit)){
             int newBoard[N][N];
             copyBoard(board, newBoard);
             newBoard[row][col] = digit;
             // Spawn a task for each valid placement.
             #pragma omp task shared(solutionFound) firstprivate(newBoard)
             {
                 backtrack(newBoard);
             }
         }
     }
     #pragma omp taskwait
 }
 
 /*
  * solveSudoku:
  *   Sets up parallel region, starts backtracking,
  *   and waits for a solution.
  */
 void solveSudoku(int board[N][N])
 {
     solutionFound = 0; // Reset before solving
     #pragma omp parallel
     {
         #pragma omp single
         {
             backtrack(board);
         }
     }
 }
 
 /*
  * printBoard:
  *   Prints the board to stdout, using '.' for empty cells.
  */
 void printBoard(int board[N][N])
 {
     for(int i = 0; i < N; i++){
         for(int j = 0; j < N; j++){
             if(board[i][j] == 0) printf(". ");
             else printf("%d ", board[i][j]);
         }
         printf("\n");
     }
 }
 
 /*
  * main:
  *   Provides a sample Sudoku puzzle and then 
  *   calls solveSudoku() to find the solution.
  */
 int main(int argc, char *argv[])
 {
     // Example puzzle: 0 means empty
     int puzzle[N][N] = {
         {5,3,0, 0,7,0, 0,0,0},
         {6,0,0, 1,9,5, 0,0,0},
         {0,9,8, 0,0,0, 0,6,0},
 
         {8,0,0, 0,6,0, 0,0,3},
         {4,0,0, 8,0,3, 0,0,1},
         {7,0,0, 0,2,0, 0,0,6},
 
         {0,6,0, 0,0,0, 2,8,0},
         {0,0,0, 4,1,9, 0,0,5},
         {0,0,0, 0,8,0, 0,7,9}
     };
 
     printf("Initial puzzle:\n");
     printBoard(puzzle);
     printf("\nSolving...\n\n");
 
     // Solve using parallel backtracking
     solveSudoku(puzzle);
 
     if(solutionFound){
         printf("Solved:\n");
         printBoard(solvedBoard);
     } else {
         printf("No solution found.\n");
     }
 
     return 0;
 }
 