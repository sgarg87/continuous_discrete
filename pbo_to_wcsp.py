import random
import numpy as np
import numpy.random as npr
import math


class Expression:

    def __init__(self, variables, functions, functions_parm, expression_coefficient):
        self.poly = 'poly'
        self.cos = 'cos'
        self.sin = 'sin'

        self.variables = variables
        self.functions = functions
        self.functions_parm = functions_parm
        self.expression_coefficient = expression_coefficient

    def __print__(self,):
        print '..................'
        expression_str = ''

        for curr_var_idx in range(self.variables.size):
            curr_var = self.variables[curr_var_idx]
            curr_var_str = 'X'+str(curr_var)
            curr_function = self.functions[curr_var_idx]
            curr_function_parm = self.functions_parm[curr_var_idx]

            if curr_function == self.poly:
                func_str = curr_var_str+'^'+'{'+str(curr_function_parm)+'}'
            else:
                func_str = curr_function + '(' + str(curr_function_parm) + '*' + curr_var_str + ')'

            if curr_var_idx == 0:
                expression_str += func_str
            else:
                expression_str += '*'+func_str

        expression_str = str(self.expression_coefficient)+'*'+'('+expression_str+')'

        print expression_str


class PseudoBooleanOptimizationToWeightedConstraintSatisfactionProblem:

    def __init__(self):

        self.poly = 'poly'
        self.cos = 'cos'
        self.sin = 'sin'

        self.min_degree_polynomial = 0.0
        self.max_degree_polynomial = 4.0

        self.min_coefficient = -1.0
        self.max_coefficient = 1.0

        self.num_var = 4
        self.prob_var_in_expression = 0.3
        self.max_vars_per_expression = self.num_var

        self.function_choices =\
            np.array(
                [
                    self.poly,
                    self.sin,
                    self.cos,
                ]
            )

        self.max_expressions_per_objective = 5
        self.discrete_var_domain_size = 5

    def sample_coefficient(self):
        return round(random.uniform(self.min_coefficient, self.max_coefficient), 2)

    def sample_exponent(self):
        return round(random.uniform(self.min_degree_polynomial, self.max_degree_polynomial), 2)

    def sample_random_monomial(self):

        # sample a function from choices
        func = npr.choice(self.function_choices)

        if func == self.poly:
            poly_degree = self.sample_exponent()
            return func, poly_degree
        else:
            coefficient = self.sample_coefficient()
            return func, coefficient

    def sample_expression(self, num_var):

        variables = []
        functions = []
        functions_parm = []

        for var in range(num_var):
            if random.random() < self.prob_var_in_expression:
                variables.append(var)
                func, func_parm = self.sample_random_monomial()
                functions.append(func)
                functions_parm.append(func_parm)

                if len(variables) == self.max_vars_per_expression:
                    break

        variables = np.array(variables)
        functions = np.array(functions)
        functions_parm = np.array(functions_parm)
        expression_coefficient = self.sample_coefficient()

        expression_obj = Expression(variables, functions, functions_parm, expression_coefficient)

        return expression_obj

    def sample_random_objective(self):

        num_expressions_per_objective = random.randint(1, self.max_expressions_per_objective)
        expression_objs = []

        for curr_expression_idx in range(num_expressions_per_objective):
            curr_expression_obj = self.sample_expression(num_var=self.num_var)

            if curr_expression_obj.variables.size == 0:
                continue
            else:
                expression_objs.append(curr_expression_obj)

        expression_objs = np.array(expression_objs)

        return expression_objs

    def compute_constraint_for_expression(self, expression_obj):

        variables = expression_obj.variables
        constraint_matrix = np.zeros([self.discrete_var_domain_size]*variables.size)
        print 'constraint_matrix.shape', constraint_matrix.shape

        for var_values in np.ndindex(constraint_matrix.shape):
            curr_expression_val = self.compute_expression(
                expression_obj=expression_obj,
                var_values=np.array(var_values),
            )
            # assert curr_expression_val >= 0.0
            constraint_matrix[var_values] = curr_expression_val

        return constraint_matrix

    def compute_expression(self, expression_obj, var_values):

        assert expression_obj.variables.size == var_values.size

        expression_val = 1.0
        for curr_var_idx in range(expression_obj.variables.size):
            curr_var_val = var_values[curr_var_idx]
            curr_var_func = expression_obj.functions[curr_var_idx]
            curr_var_func_parm = expression_obj.functions_parm[curr_var_idx]

            if curr_var_func == self.poly:
                curr_func_val = curr_var_val**curr_var_func_parm
            else:
                if curr_var_func == self.sin:
                    curr_func_val = math.sin(curr_var_func_parm*curr_var_val)
                elif curr_var_func == self.cos:
                    curr_func_val = math.cos(curr_var_func_parm*curr_var_val)
                else:
                    raise AssertionError

            expression_val *= curr_func_val

        expression_val *= expression_obj.expression_coefficient

        return expression_val

    def compute_weighted_constraints_for_objective(self, expression_objs):

        objective_const = 0.0

        for curr_expression_obj in expression_objs:
            curr_expression_obj.__print__()

            constraint_matrix = self.compute_constraint_for_expression(curr_expression_obj)
            curr_expression_obj.constraint_matrix = constraint_matrix

            # this is to ensure that constraint values are positive
            min_constraint_val_constant = min(constraint_matrix.min(), 0.0)
            curr_expression_obj.min_constraint_val_constant = min_constraint_val_constant

            print 'min_constraint_val_constant', min_constraint_val_constant

            objective_const += min_constraint_val_constant

            print '\n'

        print 'objective_const', objective_const
        self.objective_const = objective_const

    def write_weighted_constraints_for_objective(self, expression_objs):

        with open('./wcsp_dc.txt', 'w') as f:

            f.write('dc {} {} {} {}\n'.format(
                    self.num_var,
                    self.discrete_var_domain_size,
                    expression_objs.size,
                    1e100,
            ))

            f.write(' '.join([str(self.discrete_var_domain_size)]*self.num_var)+'\n')

            for curr_expression_obj in expression_objs:
                curr_expression_obj.__print__()

                constraint_matrix = curr_expression_obj.constraint_matrix
                min_constraint_val_constant = curr_expression_obj.min_constraint_val_constant

                first_line = ''
                first_line += str(curr_expression_obj.variables.size)
                for curr_var in curr_expression_obj.variables:
                    first_line += ' ' + str(curr_var)
                first_line += ' 0'
                first_line += ' '+str(constraint_matrix.size)
                f.write(first_line+'\n')

                for var_values in np.ndindex(constraint_matrix.shape):
                    curr_line = ''

                    for curr_var_val in var_values:
                        curr_line += ' ' + str(curr_var_val)

                    curr_constraint_val = (constraint_matrix[var_values] - min_constraint_val_constant)
                    curr_line += ' ' + str(curr_constraint_val)
                    curr_line = curr_line.strip()
                    f.write(curr_line+'\n')


if __name__ == '__main__':
    obj = PseudoBooleanOptimizationToWeightedConstraintSatisfactionProblem()
    expression_objs = obj.sample_random_objective()
    obj.compute_weighted_constraints_for_objective(expression_objs)
    obj.write_weighted_constraints_for_objective(expression_objs)
