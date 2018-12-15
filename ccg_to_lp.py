import numpy as np
import scipy.optimize as so
import sys
import time
import numpy.random as npr


class ConstraintsCompositeGraphToLinearProgram:

    def __init__(self):
        pass

    def extract_vertices_edges_info(self, vertex_edges_info_text):

        vertex_edges_info_arr = vertex_edges_info_text.split()

        assert vertex_edges_info_arr[0] == 'p'
        assert vertex_edges_info_arr[1] == 'edges'
        num_vertices = int(vertex_edges_info_arr[2])
        num_edges = int(vertex_edges_info_arr[3])

        return num_vertices, num_edges

    def read_vertices_weights(self, num_vertices, f):

        vertex_weights = np.zeros(num_vertices+1, dtype=np.float)

        for curr_idx in xrange(num_vertices):
            curr_vertex_info_text = f.readline()
            curr_vertex_info_arr = curr_vertex_info_text.split()
            assert curr_vertex_info_arr[0] == 'v'

            vertex_id = int(curr_vertex_info_arr[1])

            vertex_weight = float(curr_vertex_info_arr[2])
            vertex_weights[vertex_id] = vertex_weight

        return vertex_weights

    def read_edges_vertices(self, num_edges, f):

        edge_vertices = np.zeros((num_edges, 2), dtype=np.int)

        for curr_idx in xrange(num_edges):
            curr_edge_info_text = f.readline()
            curr_edge_info_arr = curr_edge_info_text.split()
            assert curr_edge_info_arr[0] == 'e'

            vertex1_id = int(curr_edge_info_arr[1])
            vertex2_id = int(curr_edge_info_arr[2])
            edge_vertices[curr_idx, 0] = vertex1_id
            edge_vertices[curr_idx, 1] = vertex2_id

        return edge_vertices

    def read_ccg(self, file_path):

        with open(file_path, 'r') as f:
            vertex_edges_info_text = f.readline()
            num_vertices, num_edges = self.extract_vertices_edges_info(vertex_edges_info_text)
            vertex_weights = self.read_vertices_weights(num_vertices=num_vertices, f=f)
            assert vertex_weights.size == (num_vertices+1)

            edge_vertices = self.read_edges_vertices(num_edges=num_edges, f=f)

        return vertex_weights, edge_vertices

    def solve_lp(self, vertex_weights, edge_vertices, lp_method):

        # vertex_weights = npr.randn(vertex_weights.size)
        # vertex_weights = np.zeros(vertex_weights.size)
        # vertex_weights[[1, 2, 3]] = 1.0
        print '-----------------------------------'
        print '-----------------------------------'
        # vertex_weights /= 10.0
        # vertex_weights += npr.randn(vertex_weights.size)
        print 'vertex_weights', vertex_weights[1:]

        num_vertices = vertex_weights.size-1
        num_edges = edge_vertices.shape[0]
        print '-----------------------------------'
        print '(v: {}, e: {})'.format(num_vertices, num_edges)

        A = np.zeros((num_edges, num_vertices))
        for curr_idx in xrange(num_edges):
            edge = edge_vertices[curr_idx]
            A[curr_idx, (edge-1)] = -1.0
        print '-----------------------------------'
        print 'A.shape', A.shape
        print 'A', A

        b = -1.0*np.ones(num_edges)
        print '-----------------------------------'
        print 'b.shape', b.shape
        print 'b', b

        start_time = time.time()
        print '-----------------------------------'
        print 'Solving the linear program ...'
        res = so.linprog(
                    c=vertex_weights[1:],
                    A_ub=A,
                    b_ub=b,
                    options={"disp": True, 'maxiter': 10000},
                    # bounds=(0.0, 1.0),
                    method=lp_method,
            )
        print 'Time to solve the program {}'.format(time.time() - start_time)
        print '-----------------------------------'
        print '-----------------------------------'

        fun = res['fun']
        x = res['x']

        return res, fun, x

    def main(self, file_path):

        vertex_weights, edge_vertices = self.read_ccg(file_path=file_path)

        # res, fun, x = self.solve_lp(vertex_weights, edge_vertices, lp_method='simplex')
        # print res

        res, fun, x = self.solve_lp(vertex_weights, edge_vertices, lp_method='interior-point')
        print res


if __name__ == '__main__':

    file_path = sys.argv[1]
    obj = ConstraintsCompositeGraphToLinearProgram()
    obj.main(file_path=file_path)
