import numpy as np
from tqdm import tqdm

class LinearEquationSolver:
    def __init__(self, A, b, epsilon=10e-6):
        self.dtype = np.float64   # affect the accuracy of the solutions (worth further studying)
        self.A = A.astype(self.dtype)
        self.b = b.astype(self.dtype)

        self.A_height = A.shape[0]
        self.A_width = A.shape[1]

        self.epsilon = epsilon
    
    def _compare_zero(self, i):    # whether zero under float implementation
        self.A[i][np.logical_and(self.A[i] < self.epsilon, self.A[i] > -self.epsilon)] = 0
        self.b[i][np.logical_and(self.b[i] < self.epsilon, self.b[i] > -self.epsilon)] = 0

    def _switch_row(self, i, j):
        # produce transition matrix
        trans_mat = np.identity(self.A_height, dtype=self.dtype)
        trans_mat[i][i] = 0
        trans_mat[i][j] = 1
        trans_mat[j][j] = 0
        trans_mat[j][i] = 1
        
        self.A =  trans_mat.dot(self.A)
        self.b =  trans_mat.dot(self.b)

    def _add_row(self, i, j, coef):     # add coef*i_row to j_row
        # produce transition matrix
        trans_mat = np.identity(self.A_height, dtype=self.dtype)
        trans_mat[j][i] = coef
        
        self.A =  trans_mat.dot(self.A)
        self.b =  trans_mat.dot(self.b)

        # self._compare_zero(j)
    
    def _inspect_col(self, col_num, which_row):
        col = self.A[:, col_num]
        non_zero_index = np.argwhere(col[which_row:] != 0)
        if len(non_zero_index) != 0:
            non_zero_row = non_zero_index[0] + which_row
            non_zero_row = non_zero_row.item()
        else:
            non_zero_row = None
        return non_zero_row
    
    def _downwards_eliminate(self, col_num, non_zero_row_num):
        for row in range(non_zero_row_num+1, self.A_height):
            coef = -self.A[row, col_num]/self.A[non_zero_row_num, col_num]
            self._add_row(non_zero_row_num, row, coef)

    def _upwards_eliminate(self, col_num, non_zero_row_num):
        for row in range(0, non_zero_row_num):
            coef = -self.A[row, col_num]/self.A[non_zero_row_num, col_num]
            self._add_row(non_zero_row_num, row, coef)

    def _self_eliminate(self, col_num, non_zero_row_num):
        coef = 1/self.A[non_zero_row_num, col_num]
        self._add_row(non_zero_row_num, non_zero_row_num, coef)

    def _cal_row_echelon_form(self):
        col_num = 0
        row_num = 0
        print("Solving linear equations...")
        pbar = tqdm(total=self.A_width, desc="might end before hitting the end")
        while col_num < self.A_width and row_num < self.A_height:
            non_zero_row = self._inspect_col(col_num, row_num)
            if non_zero_row is None:
                col_num += 1
                pbar.update(1)
                continue
            self._switch_row(row_num, non_zero_row)
            self._self_eliminate(col_num, non_zero_row)
            self._downwards_eliminate(col_num, non_zero_row)
            self._upwards_eliminate(col_num, non_zero_row)
            col_num += 1
            row_num += 1
            pbar.update(1)
    
    def _inspect_solution_existence(self):
        for row, ele_b in zip(self.A, self.b):
            if not np.any(row):     # whether all zero
                if ele_b != 0:
                    return False    # solution doesn't exist
        return True
    
    def _find_pivotal_variables(self):
        pivotal_variables = []
        for row in self.A:
            non_zero_index = np.argwhere(row != 0)
            if len(non_zero_index) == 0:
                break
            pivotal_variables.append(non_zero_index[0])
        free_variables = [i for i in range(self.A_width) if i not in pivotal_variables]
        return pivotal_variables, free_variables
    
    def _cal_null_solutions(self, pivotal_variables, free_variables):
        solutions = []
        for i in free_variables:
            v = np.zeros((self.A_width, 1), dtype=self.dtype)
            v[i, 0] = 1
            null_b = 0 - self.A.dot(v)
            for ind, p in enumerate(pivotal_variables):
                v[p, 0] = null_b[ind, 0]
            solutions.append(v.flatten())
        return solutions
    
    def solve(self):
        # calculate the row echelon form
        self._cal_row_echelon_form()
        # determine whether a solution exists
        if not self._inspect_solution_existence():
            return {"special_solution": None, "null_solution": []}
        # calculate pivotal variables
        pivotal_variables, free_variables = self._find_pivotal_variables()
        # calculate special solution
        special_solution = np.zeros((self.A_width), dtype=self.dtype)
        for i, p in enumerate(pivotal_variables):
            special_solution[p] = self.b[i]
        # calculate null solution
        null_solutions = self._cal_null_solutions(pivotal_variables, free_variables)
        # return
        sols = {"special_solution": special_solution, "null_solution": null_solutions}
        return sols
    
    def solve_and_random_pick(self, n, scope=1):     # n, how many solutions to pick
        sols = self.solutions
        if sols == "no solution exists":
            return []
        if sols["null_solution"] != 0:
            selected_sols = []
            for _ in range(n):
                null_sols = np.stack(sols["null_solution"])
                v = np.random.uniform(-scope, scope, (1, len(sols["null_solution"])))
                selected_sols.append(v.dot(null_sols).flatten() + sols["special_solution"])
            return selected_sols
        else:
            return [sols["special_solution"]]


if __name__ == "__main__":
    A = np.array([
        [0.1, 2, 1],
        [0.3, 6, 5]
    ])
    b = np.array([
        [-2],
        [-6]
    ])

    sols = LinearEquationSolver(A, b).solve()
    print(sols)

