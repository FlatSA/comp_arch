#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int rank, proc_num;
	int l, m;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0) {
		cin >> l >> m;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

	int el_per_proc = (m * l) / proc_num;
	int diff = (m * l) % proc_num;

	vector<int> proc_el_num(proc_num);
	vector<int> shift(proc_num);

	for(int i = 0; i < proc_num; i++) {
		proc_el_num[i] = el_per_proc;		
		shift[i] = i * el_per_proc;
	}

	proc_el_num[proc_num - 1] += diff;

	vector<double> a(l * m);
	vector<double> b(l * m);
	vector<double> c(l * m, 0);
	if(rank == 0) {
		srand(time(0));
		for(int i = 0; i < m * l; i++) {
			a[i] = rand() % 10 + 1;
			b[i] = rand() % 10 + 1;
		}
	}

	vector<double> buf_a(proc_el_num[rank]);
	vector<double> buf_b(proc_el_num[rank]);
	vector<double> buf_c(proc_el_num[rank]);

	MPI_Scatterv(a.data(), proc_el_num.data(), shift.data(), MPI_DOUBLE, 
	      buf_a.data(), proc_el_num[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(b.data(), proc_el_num.data(), shift.data(), MPI_DOUBLE,
	      buf_b.data(), proc_el_num[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for(int i = 0; i < proc_el_num[rank]; i++) {
		buf_c[i] = buf_a[i] + buf_b[i];
	}

	MPI_Gatherv(buf_c.data(), proc_el_num[rank], MPI_DOUBLE, c.data(),
	     proc_el_num.data(), shift.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if(rank == 0) {
		cout << "matrix a: \n";
		for(int i = 0; i < l; i++) {
			for(int j = 0; j < m; j++) {
				cout << a[i * l + j] << ' ';
			}
			cout << '\n';
		}
		cout << "matrix b: \n";
		for(int i = 0; i < l; i++) {
			for(int j = 0; j < m; j++) {
				cout << b[i * l + j] << ' ';
			}
			cout << '\n';
		}

		cout << "matrix (a + b): \n";
		for(int i = 0; i < l; i++) {
			for(int j = 0; j < m; j++) {
				cout << c[i * l + j] << ' ';
			}
			cout << '\n';
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	cout << "rank: " << rank << ", elements proccesed: " << proc_el_num[rank] << '\n';
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
