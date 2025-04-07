#include <mpi.h>
#include <vector>
#include <iostream>
#include <limits>

using namespace std;

void generate_data(vector<unsigned>& v, unsigned& n, unsigned& n_adjusted, int world_size) {
    cin >> n;

    double start_gen_time = MPI_Wtime();

    n_adjusted = (n + world_size - 1) / world_size * world_size;
    v.resize(n_adjusted);

    unsigned mul = 1664525, add = 1013904223, cur = 123456789;
    for (unsigned i = 0; i < n; i++) {
        v[i] = (cur = cur * mul + add) % 2000000001;
    }
    for (unsigned i = n; i < n_adjusted; i++) {
        v[i] = numeric_limits<unsigned>::max();
    }

    double end_gen_time = MPI_Wtime();
    cerr << (end_gen_time - start_gen_time) << " s - finished data generation\n";
}

void scatter_data(vector<unsigned>& v, vector<unsigned>& local_v, unsigned& n_adjusted, int world_size) {
    unsigned local_n;

    MPI_Bcast(&n_adjusted, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    local_n = n_adjusted / world_size;
    local_v.resize(local_n);

    MPI_Scatter(v.data(), local_n, MPI_UNSIGNED, local_v.data(), local_n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
}

void find_local_min_max(const vector<unsigned int>& local_v, unsigned int& local_min, unsigned int& local_max) {
    local_min = numeric_limits<unsigned int>::max();
    local_max = numeric_limits<unsigned int>::min();

    for (unsigned int num : local_v) {
        if (num != numeric_limits<unsigned int>::max()) {
            if (num < local_min) local_min = num;
            if (num > local_max) local_max = num;
        }
    }
}

void reduce_global_min_max(unsigned local_min, unsigned local_max, unsigned& global_min, unsigned& global_max) {
    MPI_Reduce(&local_min, &global_min, 1, MPI_UNSIGNED, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    unsigned n, n_adjusted;
    vector<unsigned> v, local_v;

    if (world_rank == 0) {
        generate_data(v, n, n_adjusted, world_size);
    }

    scatter_data(v, local_v, n_adjusted, world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_comp_time = MPI_Wtime();

    unsigned local_min, local_max, global_min, global_max;
    find_local_min_max(local_v, local_min, local_max);

    double local_end_time = MPI_Wtime();
    double local_duration = local_end_time - start_comp_time;

    cout << "Process " << world_rank << ": Local min = " << local_min
        << ", Local max = " << local_max
        << " (found in " << local_duration << " s)" << endl;

    reduce_global_min_max(local_min, local_max, global_min, global_max);

    double end_comp_time = MPI_Wtime();

    if (world_rank == 0) {
        cout << "Global minimum: " << global_min << endl;
        cout << "Global maximum: " << global_max << endl;
        cerr << (end_comp_time - start_comp_time) << " s - finished computation\n";
    }

    MPI_Finalize();
    return 0;
}
