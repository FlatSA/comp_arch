#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, proc_num;
	int k = 4;
	int l = 3;
	int m = 5;

	vector<double> a(k * l);
	vector<double> b(l * m);
	vector<double> c(k * m, 0);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank == 0) { 
		srand(time(0));
		for(int i = 0; i < k * l; i++) {
			a[i] = rand() % 3 - 1;
		}

		for(int i = 0; i < l * m; i++) {
			b[i] = rand() % 3 - 1;
		}
	}

	MPI_Bcast(b.data(), m * l, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int str_proc = k / proc_num;
	int diff = k % proc_num;

	vector<int> proc_el_num(proc_num);
	vector<int> shift(proc_num);

	vector<int> proc_el_num_back(proc_num);
	vector<int> back_shift(proc_num);

	for(int i = 0; i < proc_num; i++) {
		proc_el_num[i] = str_proc * l;
		shift[i] = proc_el_num[i] * i;

		proc_el_num_back[i] = str_proc * m;
		back_shift[i] = proc_el_num_back[i] * i;
	}

	proc_el_num[proc_num - 1] += diff * l;
	proc_el_num_back[proc_num - 1] += diff * m;

	vector<double> buf_a(proc_el_num[rank]);
	vector<double> buf_c(proc_el_num_back[rank]);

	MPI_Scatterv(a.data(), proc_el_num.data(), shift.data(), MPI_DOUBLE, 
	      buf_a.data(), proc_el_num[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int lines = proc_el_num[rank] / l;
	for(int i = 0; i < lines; i++) {
		for(int j = 0; j < m; j++) {
			double tmp = 0;

			for(int q = 0; q < l; q++) {
				tmp += buf_a[i * l + q] * b[m * q + j];
			}

			buf_c[i * m + j] = tmp;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gatherv(buf_c.data(), proc_el_num_back[rank], MPI_DOUBLE, c.data(),
	     proc_el_num_back.data(), back_shift.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(rank == 0) {
		cout << "Matrix a: \n";
		for(int i = 0; i < k; i++) {
			for(int j = 0; j < l; j++) {
				cout << a[i * l + j] << ' ';
			}
			cout << '\n';
		}

		cout << "Matrix b: \n";
		for(int i = 0; i < l; i++) {
			for(int j = 0; j < m; j++) {
				cout << b[i * m + j] << ' ';
			}
			cout << '\n';
		}

		cout << "Matrix a * b: \n";
		for(int i = 0; i < k; i++) {
			for(int j = 0; j < m; j++) {
				cout << c[i * m + j] << ' ';
			}
			cout << '\n';
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	cout << "Rank: " << rank << ", lines proccesed: " << proc_el_num[rank] / l << '\n';
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
