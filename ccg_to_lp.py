import numpy as np
import scipy.optimize as so
import sys
import time


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

    def solve_lp(self, vertex_weights, edge_vertices):

        num_vertices = vertex_weights.size-1
        num_edges = edge_vertices.shape[0]

        A = np.zeros((num_edges, num_vertices))

        for curr_idx in xrange(num_edges):
            edge = edge_vertices[curr_idx]
            A[curr_idx, (edge-1)] = -1

        b = -1*np.ones(num_edges)

        start_time = time.time()
        print 'Solving the linear program ...'
        res = so.linprog(
                    c=vertex_weights[1:],
                    A_ub=A,
                    b_ub=b,
                    options={"disp": True},
                    bounds=(0.0, 1.0)
            )
        print 'Time to solve the program {}'.format(time.time() - start_time)

        return res

    def main(self, file_path):

        vertex_weights, edge_vertices = self.read_ccg(file_path=file_path)
        res = self.solve_lp(vertex_weights, edge_vertices)
        print res


if __name__ == '__main__':

    file_path = sys.argv[1]
    obj = ConstraintsCompositeGraphToLinearProgram()
    obj.main(file_path=file_path)
