#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv) {
	int proc_rank, proc_num;
	int len = atoi(argv[1]);

	vector<double> a(len);
	vector<double> b(len);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	if(proc_rank == 0) {
		srand(time(0));
		for(int i = 0; i < len; i++) {
			a[i] = rand() % 3 - 1;
			b[i] = rand() % 3 - 1;
		}
	}

	int el_per_pr = len / proc_num;
	int diff = len % proc_num;

	vector<int> proc_el_num(proc_num);
	vector<int> proc_shift(proc_num);

	for(int i = 0; i < proc_num; i++) {
		proc_el_num[i] = el_per_pr;
		proc_shift[i] = i * el_per_pr;
	}

	proc_el_num[proc_num - 1] += diff;

	vector<double> buf_a(proc_el_num[proc_rank]);
	vector<double> buf_b(proc_el_num[proc_rank]);

	MPI_Scatterv(a.data(), proc_el_num.data(), proc_shift.data(), MPI_DOUBLE,
	      buf_a.data(), proc_el_num[proc_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(b.data(), proc_el_num.data(), proc_shift.data(), MPI_DOUBLE,
	      buf_b.data(), proc_el_num[proc_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double temp = 0;
	for(int i = 0; i < proc_el_num[proc_rank]; i++) {
		temp += buf_a[i] * buf_b[i];
	}

	double ans = 0;
	MPI_Reduce(&temp, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(proc_rank == 0) {
		cout << "vector a :";
		for(auto e : a) {
			cout << e << ' ';
		}
		cout << '\n';

		cout << "vector b: ";
		for(auto e : b) {
			cout << e << ' ';
		}
		cout << '\n';
		cout << "ans: " << ans << '\n';
	}

	MPI_Barrier(MPI_COMM_WORLD);
	cout << "proc_rank: " << proc_rank << ", elements proccesed: " << proc_el_num[proc_rank] <<
		'\n';
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
